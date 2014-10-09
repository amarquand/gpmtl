function [stats] = gp_mtr_mh(X, Y, opt)

% Subject and cross-validation parameters
[N,T] = size(Y);

%opt = check_params(opt);   % check all required parameters are specified

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MCMC Parameter Specification
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generic MCMC parameters
write_interval = round(opt.nGibbsIter/10);

% initialise top-level priors
scale_prior    = opt.PriorParam(1);  % scale matrix
nu_prior       = opt.PriorParam(2);  % degrees of freedom
a_prior        = opt.PriorParam(3);  % Parameters of prior for the noise
b_prior        = opt.PriorParam(4); 
% use raw covariance of the targets as a prior
Ym = Y - repmat(mean(Y),N,1);
YY = Ym'*Ym;
Psi_prior = scale_prior*YY;
%Psi_prior = scale_prior*eye(T);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% starting likelihood for theta
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Theta        = opt.X0_MH;
LogLik_theta = gp_mtr(Theta, X, Y, opt);
LogLik_theta = -LogLik_theta;

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% starting priors for theta
%%%%%%%%%%%%%%%%%%%%%%%%%%%
CholMask           = tril(ones(T) ~= 0); % create mask for lower diagonal
lf                 = Theta(1:nnz(CholMask));
Lf                 = zeros(T);
Lf(CholMask)       = lf;
Kf                 = Lf*Lf';
S2                 = Theta(nnz(CholMask)+1:end);
LogPrior_noise_all = zeros(length(S2));
for i = 1:length(S2)
    LogPrior_noise_all = log(gampdf(exp(-S2(i)),a_prior,1/b_prior));
end
LogPrior_noise   = sum(LogPrior_noise_all);      % - because of inverse
[~, LogPrior_Kf] = invwishpdf(Kf,Psi_prior,2*nu_prior+1);
LogPrior_theta   = LogPrior_Kf + LogPrior_noise;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Metropolis proposal distribution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Noise_proposal = chol(opt.mh.StepSize*eye(T))';
w_proposal     = opt.mh.ProposalScale;
Psi_proposal   = w_proposal*Kf;
nu_proposal    = w_proposal+T+1;

% initialize posteriors
Theta_all = zeros(size(Theta,1),opt.nGibbsIter);

% initialize stats
stats.iter           = 1;
stats.opt            = opt;
stats.prior_theta    = {Psi_prior, nu_prior};
stats.arate_mh       = zeros(1,opt.nGibbsIter);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Begin Gibbs Sampling Block
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
acc_theta_all = 0;  gidx = 1:50; 
for g = 1:opt.nGibbsIter
    % display output
    if mod(g,50) == 0
        arate_theta = acc_theta_all / 50;
        
        disp(['Gibbs iter: ',num2str(g),' arate(theta)=',num2str(arate_theta,'%2.2f')]);
        acc_theta_all = 0; 
        
        % update stats
        stats.iter                 = g;
        stats.arate_mh(gidx)       = arate_theta;        gidx = gidx + 50;
    end 
   
    % save output
    if mod(g,write_interval) == 0 && opt.WriteInterim && ...
       isfield(opt,'OutputFilename') && ...
       ~isempty(opt.OutputFilename)
        save([opt.OutputFilename,'stats'],'stats');
        save([opt.OutputFilename,'Theta_all'],'Theta_all','-v7.3');
    end

    % sample theta
    %%%%%%%%%%%%%%
    % whiten f
    %nu = (L_Kt\f);
    
    % sample Kf
    % draw a covariance matrix (Kf), parameterised by a covariance matrix
    Kf_new = iwishrnd(Psi_proposal,nu_proposal);
    Lf_new = chol(Kf_new)';
    lf_new = Lf_new(CholMask ~= 0);      % extract lower diagonal
    
    % sample noise 
    S2_new = S2 + opt.mh.StepSize*(Noise_proposal*randn(T,1));    

    % make a step in Theta
    Theta_new = [lf_new; S2_new];
 
    % compute new likelihood
    LogLik_theta_new = gp_mtr(Theta_new, X, Y, opt);
    LogLik_theta_new = -LogLik_theta_new;
    
    % compute new priors for the noise    
    LogPrior_noise_new_all = zeros(length(S2));
    for i = 1:length(S2)
        LogPrior_noise_new_all = log(gampdf(exp(-S2_new(i)),a_prior,1/b_prior));
    end
    LogPrior_noise_new = sum(LogPrior_noise_new_all);
    
    % compute new priors for Kf
    [~, LogPrior_Kf_new] = invwishpdf(Kf_new,Psi_prior,2*nu_prior+1);

    % Final priors for theta
    LogPrior_theta_new = LogPrior_Kf_new + LogPrior_noise_new;

    HastingsRatio = 0.5*(2*w_proposal+3*T+3)*log(det(Kf)) + ...
                    0.5*w_proposal*(trace(Kf_new\Kf) -trace(Kf\Kf_new)) - ...
                    0.5*(2*w_proposal+3*T+3)*log(det(Kf_new));
    
    Ratio = LogLik_theta_new + LogPrior_theta_new + HastingsRatio - (LogLik_theta + LogPrior_theta);
    if Ratio > 0 || (Ratio > log(rand)) % accept
        Theta          = Theta_new;
        Kf             = Kf_new;
        S2             = S2_new;
        %LogPrior_Kf    = LogPrior_Kf_new;
        %LogPrior_noise = LogPrior_noise_new;        
        LogPrior_theta = LogPrior_theta_new;
        
        % update Metropolis proposal distribution
        Psi_proposal   = w_proposal*Kf;
        
        acc_theta = 1;
    else % reject
        acc_theta = 0;
    end
    
    Theta_all(:,g) = Theta;
    
    %if norm(Theta) ~= norm(Theta_new), acc_theta = 1; else acc_theta = 0; end
    acc_theta_all = acc_theta_all + acc_theta;
    
    if g == opt.BurnIn
        tic; % start timer
    end
end
stats.time_taken       = toc;
stats.arate_theta_mean = mean(stats.arate_mh);

disp(['Mean acceptance rate (theta): ',num2str(stats.arate_theta_mean,'%2.2f')]);

if isfield(opt,'OutputFilename') && ~isempty(opt.OutputFilename)
    save([opt.OutputFilename,'Theta_all'],'Theta_all','-v7.3');
    %save([opt.OutputFilename,'f_all'],'f_all','-v7.3');    
    save([opt.OutputFilename,'stats'],'stats','-v7.3');
end
end

