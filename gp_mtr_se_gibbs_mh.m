function [stats] = gp_mtr_se_gibbs_mh(X, Y, opt)

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
a_noise        = opt.PriorParam(3);  % Parameters of prior for the noise
b_noise        = opt.PriorParam(4); 
a_ell          = opt.PriorParam(5);  % Parameters of prior for the noise
b_ell          = opt.PriorParam(6); 
a_sf2          = opt.PriorParam(7);  % Parameters of prior for the noise
b_sf2          = opt.PriorParam(8); 
if opt.UseYYPrior
    disp('Using informed task prior')
    Ym = Y - repmat(mean(Y),N,1);
    YY = Ym'*Ym;
    Psi_prior = scale_prior*YY;
else
    disp('Using isotropic task prior')
    Psi_prior = scale_prior*eye(T);
end

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
ell                = Theta(nnz(CholMask)+1);
sf2                = Theta(nnz(CholMask)+2);
S2                 = Theta(nnz(CholMask)+3:end);
%LogPrior_se_all    = zeros(2,1);
LogPrior_noise_all = zeros(length(S2),1);

LogPrior_sf2 = a_sf2*log(b_sf2) - gammaln(a_sf2) - (a_sf2+1)*exp(sf2) - b_sf2/exp(sf2);    
LogPrior_ell = log(gampdf(exp(ell),a_ell,1/b_ell)); 

for i = 1:length(S2)
    % true inverse Gamma (not a shortcut!)
    LogPrior_noise_all(i) = a_noise*log(b_noise) - gammaln(a_noise) - ...
                            (a_noise+1)*exp(S2(i)) - b_noise/exp(S2(i));
end
LogPrior_noise   = sum(LogPrior_noise_all); 

[~, LogPrior_Kf] = invwishpdf(Kf,Psi_prior,2*nu_prior+1);
LogPrior_theta   = LogPrior_Kf + LogPrior_sf2 + LogPrior_ell + LogPrior_noise;

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
stats.arate_noise    = zeros(1,opt.nGibbsIter);
stats.arate_Kf       = zeros(1,opt.nGibbsIter);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Begin Gibbs Sampling Block
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
acc_noise_all = 0; acc_sf2_all = 0; acc_ell_all = 0;acc_Kf_all = 0; gidx = 1:50; 
for g = 1:opt.nGibbsIter
    % display output
    if mod(g,50) == 0
        arate_noise = acc_noise_all / 50;
        arate_sf2   = acc_sf2_all / 50;
        arate_ell   = acc_ell_all / 50;
        arate_Kf    = acc_Kf_all / 50;
        
        disp(['Gibbs iter: ',num2str(g),...
              ' arate(sf2)=',num2str(arate_sf2,'%2.2f'),...
              ' arate(ell)=',num2str(arate_ell,'%2.2f'),...
              ' arate(noise)=',num2str(arate_noise,'%2.2f'),...
              ' arate(Kf)=',num2str(arate_Kf,'%2.2f')]);
        acc_noise_all = 0; 
        acc_Kf_all    = 0; 
        acc_sf2_all   = 0;
        acc_ell_all   = 0;
          
        % update stats
        stats.iter                 = g;
        stats.arate_noise(gidx)    = arate_noise;   
        stats.arate_Kf(gidx)       = arate_Kf; 
        gidx = gidx + 50;
    end 
   
    % save output
    if mod(g,write_interval) == 0 && opt.WriteInterim && ...
       isfield(opt,'OutputFilename') && ...
       ~isempty(opt.OutputFilename)
        save([opt.OutputFilename,'stats'],'stats');
        save([opt.OutputFilename,'Theta_all'],'Theta_all','-v7.3');
    end

    % sample scale parameter
    %%%%%%%%%%%%%%%%%%%%%%%%  
    sf2_new            = sf2 + 1*opt.mh.StepSize*(randn(1)); 
    LogPrior_sf2_new   = log(gampdf(exp(sf2_new),a_sf2,1/b_sf2));
    LogPrior_theta_new = LogPrior_Kf + LogPrior_ell + LogPrior_sf2_new + LogPrior_noise;
    
    Theta_new        = [lf; ell; sf2_new; S2];
  
    % compute new likelihood
    try
        LogLik_theta_new = gp_mtr(Theta_new, X, Y, opt);    
        LogLik_theta_new = -LogLik_theta_new;
        sing_K = false;
    catch
        disp('invalid covariance (sf2)')
        sing_K = true;
    end
    
    % Accept / reject for sf2
    Ratio = LogLik_theta_new + LogPrior_Kf + LogPrior_theta_new - (LogLik_theta + LogPrior_Kf + LogPrior_theta);
    if ~sing_K && (Ratio > 0 || (Ratio > log(rand))) % accept
        Theta          = Theta_new;
        sf2            = sf2_new;
        LogPrior_sf2   = LogPrior_sf2_new;
        LogPrior_theta = LogPrior_theta_new;
        LogLik_theta   = LogLik_theta_new;
        
         acc_sf2 = 1;
    else % reject
        acc_sf2 = 0;
    end
    
    % sample bandwidth parameter
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    %sf2_new = sf2 + 1*opt.mh.StepSize*(randn(1)); 
    ell_new = ell + 1*opt.mh.StepSize*(randn(1));
    
    % make a step in Theta
    Theta_new = [lf; ell_new; sf2; S2];
     
    % compute new likelihood
    try
        LogLik_theta_new = gp_mtr(Theta_new, X, Y, opt);    
        LogLik_theta_new = -LogLik_theta_new;
        sing_K = false;
    catch
        disp('invalid covariance (ell)')
        sing_K = true;
    end
    
    LogPrior_ell_new   = log(gampdf(exp(ell_new),a_ell,1/b_ell));
    LogPrior_theta_new = LogPrior_Kf + LogPrior_ell_new + LogPrior_sf2 + LogPrior_noise;
    
    % Accept / reject for SE params
    Ratio = LogLik_theta_new + LogPrior_Kf + LogPrior_theta_new - (LogLik_theta + LogPrior_Kf + LogPrior_theta);
    if ~sing_K && (Ratio > 0 || (Ratio > log(rand))) % accept
        Theta          = Theta_new;
        ell            = ell_new;
        LogPrior_ell   = LogPrior_ell_new;
        LogPrior_theta = LogPrior_theta_new;
        LogLik_theta   = LogLik_theta_new;
        
        acc_ell = 1;
    else % reject
        acc_ell = 0;
    end
    
    % sample noise 
    %%%%%%%%%%%%%%
    S2_new = S2 + opt.mh.StepSize*(Noise_proposal*randn(T,1));   
    
    % make a step in Theta
    Theta_new = [lf; ell; sf2; S2_new];
    
    % compute new likelihood
    try
        LogLik_theta_new = gp_mtr(Theta_new, X, Y, opt);    
        LogLik_theta_new = -LogLik_theta_new;
        sing_K = false;
    catch
        disp('invalid covariance (noise)')
        sing_K = true;
    end
    % compute new priors for the noise
    LogPrior_noise_new_all = zeros(length(S2),1);
    
    for i = 1:length(S2)
        LogPrior_noise_new_all(i) = a_noise*log(b_noise) - gammaln(a_noise) - ...
            (a_noise+1)*exp(S2_new(i)) - b_noise/exp(S2_new(i));
    end
    LogPrior_noise_new = sum(LogPrior_noise_new_all);
    LogPrior_theta_new = LogPrior_Kf + LogPrior_ell + LogPrior_sf2 +  LogPrior_noise_new;
    
    % Accept / reject for Theta
    Ratio = LogLik_theta_new + LogPrior_Kf + LogPrior_theta_new - (LogLik_theta + LogPrior_Kf + LogPrior_theta);
    if ~sing_K && (Ratio > 0 || (Ratio > log(rand))) % accept
        Theta          = Theta_new;
        S2             = S2_new;
        LogPrior_noise = LogPrior_noise_new;
        LogPrior_theta = LogPrior_theta_new;
        LogLik_theta   = LogLik_theta_new;
                
        acc_noise = 1;
    else % reject
        acc_noise = 0;
    end
 
    % sample Kf
    %%%%%%%%%%%
    % draw a covariance matrix (Kf), parameterised by a covariance matrix
    Kf_new = iwishrnd(Psi_proposal,nu_proposal);
    Lf_new = chol(Kf_new)';
    lf_new = Lf_new(CholMask ~= 0);      % extract lower diagonal
    
    Theta_new = [lf_new; ell; sf2; S2];
    
    % compute new likelihood
    try
        LogLik_theta_new = gp_mtr(Theta_new, X, Y, opt);
        LogLik_theta_new = -LogLik_theta_new;
        sing_K = false;
    catch
        disp('invalid covariance (Kf)')
        sing_K = true;
    end
    
    % compute new priors for Kf
    [~, LogPrior_Kf_new] = invwishpdf(Kf_new,Psi_prior,2*nu_prior+1);

    % Final priors for theta
    LogPrior_theta_new = LogPrior_Kf_new + LogPrior_ell + LogPrior_sf2 + LogPrior_noise;
        
    HastingsRatio = 0.5*(2*w_proposal+3*T+3)*log(det(Kf)) + ...
                    0.5*w_proposal*(trace(Kf_new/Kf) -trace(Kf/Kf_new)) - ...
                    0.5*(2*w_proposal+3*T+3)*log(det(Kf_new));
                
    Ratio = LogLik_theta_new + LogPrior_theta_new + HastingsRatio - (LogLik_theta + LogPrior_theta);
    if Ratio > 0 || (Ratio > log(rand)) % accept
        Theta          = Theta_new;
        lf             = lf_new;
        Kf             = Kf_new;
        LogPrior_Kf    = LogPrior_Kf_new;      
        LogPrior_theta = LogPrior_theta_new;
        %LogLik_theta   = LogLik_theta_new;
        
        % update Metropolis proposal distribution
        Psi_proposal   = w_proposal*Kf;
        
        acc_Kf = 1;
    else % reject
        acc_Kf = 0;
    end
    
    Theta_all(:,g) = Theta;
    
    acc_noise_all = acc_noise_all + acc_noise;
    acc_Kf_all    = acc_Kf_all + acc_Kf;
    acc_sf2_all   = acc_sf2_all + acc_sf2;
    acc_ell_all   = acc_ell_all + acc_ell;
    
    if g == opt.BurnIn
        tic; % start timer
    end
end
stats.time_taken       = toc;
stats.arate_noise_mean = mean(stats.arate_noise);
stats.arate_Kf_mean = mean(stats.arate_Kf);

disp(['Mean acceptance rate (noise): ',num2str(stats.arate_noise_mean,'%2.2f')]);
disp(['Mean acceptance rate (Kf): ',num2str(stats.arate_Kf_mean,'%2.2f')]);

if isfield(opt,'OutputFilename') && ~isempty(opt.OutputFilename)
    save([opt.OutputFilename,'Theta_all'],'Theta_all','-v7.3');
    %save([opt.OutputFilename,'f_all'],'f_all','-v7.3');    
    save([opt.OutputFilename,'stats'],'stats','-v7.3');
end
end

