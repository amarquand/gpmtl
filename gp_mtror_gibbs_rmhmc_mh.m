function [stats] = gpc_or_gibbs_rmhmc_mh(X, Y, opt)

% Subject and cross-validation parameters
[n,k] = size(Y);
n_gps = length(opt.X0_RMHMC)/n;
%n_gps      = k-1;

opt = check_params(opt);   % check all required parameters are specified

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MCMC Parameter Specification
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generic MCMC parameters
write_interval = round(opt.nGibbsIter/10);
a              = opt.PriorParam(1);
b              = opt.PriorParam(2);

% starting estimates
startidx= []; % we may not be using all the gps (e.g. k-1 latent functions
for c = 1:n_gps;
    startidx = [startidx (1:n)+(c-1)*n];
end
f        = opt.X0_RMHMC(startidx);
LogTheta = opt.X0_MH;

% initialize posteriors
f_all        = zeros(size(f,1),opt.nGibbsIter);   fidx = 1;
LogTheta_all = zeros(size(LogTheta,1),opt.nGibbsIter);

% Metropolis proposal distribution
MH_Proposal = eye(length(LogTheta)); 
L_Proposal  = chol(MH_Proposal)';

% initialize stats
stats.iter           = 1;
stats.opt            = opt;
stats.prior_theta    = [a, b];
stats.arate_mh       = zeros(1,opt.nGibbsIter);
stats.arate_rmhmc    = zeros(1,opt.nGibbsIter);
stats.failrate_rmhmc = zeros(1,opt.nGibbsIter);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Begin Gibbs Sampling Block
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute & invert K
% LogTheta(1) = latent scale, Logtheta(2) = likelihood variance
K = feval(opt.CovFunc,X,LogTheta(1)); 

ridge   = 1e-1+eye(n);
InvK    = inv(K+ridge);
L_Kt    = chol(K+ridge)';
LogDetK = 2*sum(log(diag(L_Kt)));
%LogDetK = log(det(K+ridge));

acc_theta_all   = 0; acc_f_all = 0; gidx = 1:50; fail_f_all = 0;
for g = 1:opt.nGibbsIter
    if mod(g,50) == 0
        arate_f     = acc_f_all / 50;
        arate_theta = acc_theta_all / 50;
        fail_f      = fail_f_all / 50;
        
        disp(['Gibbs iter: ',num2str(g),' arate(f)=',num2str(arate_f,'%2.2f'),' arate(theta)=',num2str(arate_theta,'%2.2f'),' fail(f)=',num2str(fail_f,'%2.2f'),  ]);
        acc_theta_all = 0; acc_f_all = 0; fail_f_all = 0;
        
        % update stats
        stats.iter                 = g;
        stats.arate_mh(gidx)       = arate_theta;
        stats.arate_rmhmc(gidx)    = arate_f;
        stats.failrate_rmhmc(gidx) = fail_f_all;
        gidx = gidx + 50;
    end
    % save output
    if mod(g,write_interval) == 0 && opt.WriteInterim && ...
       isfield(opt,'OutputFilename') && ...
       ~isempty(opt.OutputFilename)
        save([opt.OutputFilename,'stats'],'stats');
        save([opt.OutputFilename,'f_all'],'f_all','-v7.3');
        save([opt.OutputFilename,'LogTheta_all'],'LogTheta_all','-v7.3');
    end
    
    % sample f
    %%%%%%%%%% 
    gxargs_f = {InvK, Y};
    fxargs_f = {Y, InvK, LogDetK, LogTheta(2)};
    if opt.UseRMHMC
        error('RMHMC not implemented yet');
        %try
        %    f_new = rmhmc(f, 'rmhmc_poiss_posterior_f', fxargs_f,...
        %        'rmhmc_compute_G_f', gxargs_f, opt.rmhmc);
        %    f_new = f_new'; % transpose (for consistency)
        %catch
        %    fail_f_all = fail_f_all + 1;
        %    f_new = f;
        %end
    else
        if opt.UseGMassForHMC
            Gf = InvK;
        else
            Gf = eye(length(f));
        end
        L_Gf        = chol(Gf)';
        InvGf       = inv(Gf);
        [Ef, f_new] = hmc(f, 'hmc_negbinom_posterior_f', opt.rmhmc,  L_Gf, InvGf, fxargs_f{:});           
        f_new       = f_new(:,end); % just take the last sample (for consistency)
    end
    % test acceptance
    if norm(f) ~= norm(f_new), acc_f = 1; else acc_f = 0; end
    
    % update f
    f         = f_new;
    acc_f_all = acc_f_all + acc_f;
        
    % sample theta
    %%%%%%%%%%%%%%    
    if opt.OptimiseTheta
        % whiten f
        nu = (L_Kt\f);
        
        % make a step in theta
        LogTheta_new = LogTheta + opt.mh.StepSize*(L_Proposal*randn(length(LogTheta),1));
        
        % compute new K and invert it
        K_new       = feval(opt.CovFunc, X, LogTheta_new(1));
        
        InvK_new    = inv(K_new+ridge);
        L_Kt_new    = chol(K_new+ridge)';
        LogDetK_new = 2*sum(log(diag(L_Kt_new)));
        %LogDetK_new = log(det(K_new+ridge));
        
        f_new = L_Kt_new*nu;
        
        % compute p(y|f) and p(y|f')
        [tmp, LogLik_f]     = likelihood_negbinomial(f,Y,LogTheta(2));
        [tmp, LogLik_f_new] = likelihood_negbinomial(f_new,Y,LogTheta_new(2));
        
        % compute p(theta) and p(theta')
        LogPrior_theta_all     = zeros(size(LogTheta));
        LogPrior_theta_new_all = zeros(size(LogTheta));
        const                  = a*log(b) - gammaln(a);
        for i = 1:length(LogPrior_theta_all)
            LogPrior_theta_all(i)     = const + (a - 1)*LogTheta(i) - b*exp(LogTheta(i));
            LogPrior_theta_new_all(i) = const + (a - 1)*LogTheta_new(i) - b*exp(LogTheta_new(i));
        end
        LogPrior_theta     = sum(LogPrior_theta_all);
        LogPrior_theta_new = sum(LogPrior_theta_new_all);
        
        Ratio = (LogLik_f_new + LogPrior_theta_new) - (LogLik_f + LogPrior_theta);
        
        if Ratio > 0 || (Ratio > log(rand)) % accept
            % update theta
            LogTheta  = LogTheta_new;
            acc_theta = 1;
            
            % update f
            f       = f_new;
            InvK    = InvK_new;
            LogDetK = LogDetK_new;
            L_Kt    = L_Kt_new;
        else % reject
            acc_theta = 0;
        end
        
        LogTheta_all(:,g) = LogTheta;
        
        %if norm(LogTheta) ~= norm(LogTheta_new), acc_theta = 1; else acc_theta = 0; end
        acc_theta_all = acc_theta_all + acc_theta;
    end
    
    f_all(:,fidx) = f;
    fidx = fidx + 1;
    
    if g == opt.BurnIn
        tic; % start timer
    end
end
stats.time_taken       = toc;
stats.arate_f_mean     = mean(stats.arate_rmhmc);
stats.arate_theta_mean = mean(stats.arate_mh);

disp(['Mean acceptance rate (f): ',num2str(stats.arate_f_mean,'%2.2f')]);
disp(['Mean acceptance rate (theta): ',num2str(stats.arate_theta_mean,'%2.2f')]);

% % Exclude burn-in samples
% if isfield(opt,'BurnIn')
%     srange = opt.BurnIn+1:length(f_all);
% else 
%     srange = 1:length(f_all);
% end
% f_post        = f_all(:,srange); % ./ repmat( sum(f_all(:,MCMC_range),2), 1, length(MCMC_range));
% LogTheta_post = LogTheta_all(:,srange);

if isfield(opt,'OutputFilename') && ~isempty(opt.OutputFilename)
    %save([opt.OutputFilename,'f_post'],'f_post','-v7.3');
    %save([opt.OutputFilename,'LogTheta_post'],'LogTheta_post','-v7.3');
    save([opt.OutputFilename,'LogTheta_all'],'LogTheta_all','-v7.3');
    save([opt.OutputFilename,'f_all'],'f_all','-v7.3');    
    save([opt.OutputFilename,'stats'],'stats','-v7.3');
end
end

