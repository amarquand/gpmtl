function [stats] = gp_mth_gibbs(X, Y, opt)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Basic parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if iscell(Y) % non-block design
    N = length(Y{1});
    T = size(X{2},2);   
    Ys = X{2};
    y  = Y{1};
else
    [N,T] = size(Y);  
end

Kx = X{1};

% id matrices to keep track of everything.
Tid = X{2}; V = X{3};  Rid = X{4};
rid = logical(V*Rid(:)); %tid = logical(V*Tid(:)); 
Tid_all = zeros(N*T,T); Rid_all = zeros(N*T,T);
for t = 1:T
    Tid_all(N*(t-1)+(1:N),t) = Tid(:,t);
    Rid_all(N*(t-1)+(1:N),t) = Rid(:,t);
end    
Tid_all  = V*Tid_all;
Rid_all  = V*Rid_all;

f     = opt.X0_f;
Theta = opt.X0_Theta;
y     = Y(:); y(isnan(y)) = 0;
y     = V*y;

% hyperparameters
CholMask           = tril(ones(T) ~= 0); % create mask for lower diagonal
lf                 = Theta(1:nnz(CholMask));
Lf                 = zeros(T);
Lf(CholMask)       = lf;
Kf                 = Lf*Lf';
S2                 = Theta(nnz(CholMask)+1:end);
Noise              = zeros(T,1); 
Noise(Rid(1,:)~=0) = S2;%exp(2*LogS2); 
Noise = diag((Tid_all.*Rid_all)*Noise);

%opt = check_params(opt);   % check all required parameters are specified

RandStream.setGlobalStream(RandStream('mt19937ar','seed',sum(100*clock)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MCMC Parameter Specification
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generic MCMC parameters
write_interval = round(opt.nGibbsIter/10);

% initialise top-level priors
Psi_prior   = opt.PriorParam{1};  % scale matrix
nu_prior    = opt.PriorParam{2};  % degrees of freedom
a_prior     = opt.PriorParam{3};  % Parameters of prior for the noise
b_prior     = opt.PriorParam{4}; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% starting likelihood for theta
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%LogLik_theta = gp_mtr(Theta, X, Y, opt);
%LogLik_theta = -LogLik_theta;

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% starting priors for theta
%%%%%%%%%%%%%%%%%%%%%%%%%%%
LogPrior_noise_all = zeros(length(S2),1);
for i = 1:length(S2) % inverse Gamma 
    LogPrior_noise_all(i) = a_prior*log(b_prior) - gammaln(a_prior) - ...
                            (a_prior+1)*S2(i) - b_prior/S2(i);                   
end
LogPrior_noise   = sum(LogPrior_noise_all); 

% in the following Kf ~ IW(Kf|Psi_prior,nu_prior), but the implementation
% uses a reparameterisation: Kf ~ IW(Kf|Psi_prior,b), b = nu_prior + p + 1
[~, LogPrior_Kf] = invwishpdf(Kf,Psi_prior,nu_prior+T+1);
LogPrior_theta   = LogPrior_Kf + LogPrior_noise;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization of posteriors and stats
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialize posteriors
f_all      = zeros(size(f,1),opt.nGibbsIter);   fidx = 1;
alpha_all  = zeros(size(f,1),opt.nGibbsIter); % saves time for regression tasks
Theta_all  = zeros(size(Theta,1),opt.nGibbsIter);

% initialize stats
stats.iter           = 1;
stats.opt            = opt;
stats.prior_theta    = {Psi_prior, nu_prior};
stats.arate_noise    = zeros(1,opt.nGibbsIter);
stats.arate_Kf       = zeros(1,opt.nGibbsIter);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Begin Gibbs Sampling Block
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
acc_noise_all = 0; acc_f_all = 0; acc_Kf_all = 0; gidx = 1:50; Mu = [];
for g = 1:opt.nGibbsIter
    % display output
    if mod(g,50) == 0
        arate_f     = acc_f_all / 50;
        arate_noise = acc_noise_all / 50;
        arate_Kf    = acc_Kf_all / 50;
        
        %disp(['Gibbs iter: ',num2str(g),' arate(noise)=',num2str(arate_noise,'%2.2f'),' arate(Kf)=',num2str(arate_Kf,'%2.2f')]);
        disp(['Gibbs iter: ',num2str(g),' arate(f)=',num2str(arate_f,'%2.2f')]);
        acc_noise_all = 0; acc_f_all = 0; acc_Kf_all    = 0; 
        
        % update stats
        stats.iter                 = g;
        stats.arate_noise(gidx)    = arate_noise;   
        stats.arate_Kf(gidx)       = arate_Kf; 
        stats.arate_rmhmc(gidx)    = arate_f;
        gidx = gidx + 50;
    end 
   
    % save output
    if mod(g,write_interval) == 0 && opt.WriteInterim && ...
       isfield(opt,'OutputFilename') && ...
       ~isempty(opt.OutputFilename)
        save([opt.OutputFilename,'stats'],'stats');        
        save([opt.OutputFilename,'f_all'],'f_all','-v7.3');
        save([opt.OutputFilename,'alpha_all'],'alpha_all','-v7.3');
        save([opt.OutputFilename,'Theta_all'],'Theta_all','-v7.3');
    end
   
    % compute covariances
    K  = feval(opt.CovFunc,X,Theta);
    Kc = K(~rid,~rid);
    Kr = K(rid,rid);
 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % sample f (classification)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    f_new = zeros(size(f));
    
    % Conditional prior. This can be done faster using partitioned inverses
    CondPrior_muc = K(~rid,rid)/Kr*f(rid); 
    CondPrior_Kc  = Kc - K(~rid,rid)/Kr*K(~rid,rid)';
    %Mu = [Mu CondPrior_muc];
    
    InvKc    = inv(CondPrior_Kc);
    L_Kc     = chol(CondPrior_Kc)';
    LogDetKc = 2*sum(log(diag(L_Kc)));
    Yc       = [y(~rid) 1-y(~rid)];
    fc       = f(~rid)-CondPrior_muc;
        
    gxargs_f = {InvKc, Yc};
    fxargs_f = {Yc, InvKc, LogDetKc};
    %fxargs_f = {Yc, InvKc, LogDetKc, CondPrior_muc};
    if opt.UseRMHMC
        error('RMHMC is not implemented for this problem');
    else
        if opt.UseGMassForHMC
            Gf = feval('hmc_compute_G_f_fixedW', fc, gxargs_f{:});
        else
            Gf = eye(length(f));
        end
        L_Gf        = chol(Gf)';
        InvGf       = inv(Gf);
        [Ef, fc_new] = hmc(fc, 'hmc_posterior_f', opt.rmhmc,  L_Gf, InvGf, fxargs_f{:});           
        fc_new       = fc_new(:,end); % just take the last sample 
    end
    % test acceptance
    if norm(fc) ~= norm(fc_new), acc_f = 1; else acc_f = 0; end
    f_new(~rid) = fc_new(1:sum(~rid));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % sample f (regression)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    % conditional prior
    CondPrior_mur = (K(rid,~rid)/Kc)*f_new(~rid);
    CondPrior_Kr  = Kr - (K(rid,~rid)/Kc)*K(~rid,rid);
    
    % exact conditional posterior
    L_Kr       = chol(CondPrior_Kr + Noise(rid,rid))';       
    alpha      = solve_chol(L_Kr',y(rid)-CondPrior_mur);
    mur_post   = CondPrior_mur + CondPrior_Kr*alpha;
    v          = L_Kr\(CondPrior_Kr');
    Sr_post    = CondPrior_Kr - v'*v;
    
    % sample from posterior
    L_Sr       = chol(Sr_post)';
    f_new(rid) = mur_post + L_Sr*randn(sum(rid),1);
    
    % update f 
    f         = f_new;
    acc_f_all = acc_f_all + acc_f;  
           
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % sample noise
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
    Regid = logical(Tid_all.*Rid_all); Regid = Regid(:,sum(Regid) ~= 0);
    S2_new = zeros(size(S2));
    for j = 1:size(Regid,2) % loop over regression tasks
        yj = y(Regid(:,j));
        fj = f(Regid(:,j));
        
        a = a_prior + 0.5*N;
        b = b_prior + 0.5*sum(yj - fj).^2;
        
        % Draw from inverse gamma
        S2_new(j) = 1./gamrnd(a,1/b);
    end 
    S2 = S2_new;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % sample task covariance
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % whiten f
    %fw = (L_Kt\f); 
    
    F    = reshape(V'*f,size(Y));
    %F    = reshape(V'*fw,size(Y));
    nu   = nu_prior + N; 
    L_Kx = chol(Kx+1e-5*eye*(size(Kx,1)))';
    %Psi  = Psi_prior + F'/Kx*F;
    Psi  = Psi_prior + F'*solve_chol(L_Kx',F);
    
    try
        Kf_new = iwishrnd(Psi,nu);
        Lf_new = chol(Kf_new)';
        Singular_Kf = false;
    catch
        Lf_new = Lf;
        Singular_Kf = true;
        disp('singular Kf')
        Kf_new = Kf;
    end
   
    % update theta
    Kf             = Kf_new;
    lf_new         = Lf_new(CholMask ~= 0);      % extract lower diagonal
    Theta_new      = [lf_new; S2];
    Theta          = Theta_new;
    
    % save posteriors and kernel weights for regression
    Theta_all(:,g)      = Theta;
    f_all(:,fidx)       = f;
    alpha_all(rid,fidx) = alpha;
    fidx                = fidx + 1;
    
    if g == opt.BurnIn
        tic; % start timer
    end
end
stats.time_taken       = toc;
stats.arate_f_mean     = mean(stats.arate_rmhmc);

disp(['Mean acceptance rate (f): ',num2str(stats.arate_f_mean,'%2.2f')]);

if isfield(opt,'OutputFilename') && ~isempty(opt.OutputFilename)
    save([opt.OutputFilename,'Theta_all'],'Theta_all','-v7.3');
    save([opt.OutputFilename,'f_all'],'f_all','-v7.3');  
    save([opt.OutputFilename,'alpha_all'],'alpha_all','-v7.3');
    save([opt.OutputFilename,'stats'],'stats','-v7.3');
end
end

