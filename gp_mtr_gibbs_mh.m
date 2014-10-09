function [stats] = gp_mtr_gibbs_mh(X, Y, opt)

% Subject and cross-validation parameters
if iscell(Y) % non-block design
    N = length(Y{1});
    T = size(X{2},2);
    
    Ys = X{2};
    y  = Y{1};
else
    [N,T] = size(Y);  
end

%opt = check_params(opt);   % check all required parameters are specified

RandStream.setGlobalStream(RandStream('mt19937ar','seed',sum(100*clock)));

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
if opt.UseYYPrior
    disp('Using informed task prior')
    Phi = X{1};
    if iscell(Y)
        YY = pinv(Ys)*(y*y' ./ Phi)*pinv(Ys)';
        YY = YY+1e-2*eye(size(YY));
    else
        Ym = Y - repmat(mean(Y),N,1);
        
        YY = 1/N * Ym'/(Phi+1e-3*eye(size(Phi)))*Ym;
    end
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
S2                 = Theta(nnz(CholMask)+1:end);
LogPrior_noise_all = zeros(length(S2),1);
for i = 1:length(S2)
    % true inverse Gamma (not a shortcut!)
    LogPrior_noise_all(i) = a_prior*log(b_prior) - gammaln(a_prior) - ...
                            (a_prior+1)*exp(S2(i)) - b_prior/exp(S2(i));                   
end
LogPrior_noise   = sum(LogPrior_noise_all);      % - because of inverse (Gamma on the precision)

% in the following Kf ~ IW(Kf|Psi_prior,nu_prior), but the implementation
% uses a reparameterisation: Kf ~ IW(Kf|Psi_prior,b), b = nu_prior + p + 1
[~, LogPrior_Kf] = invwishpdf(Kf,Psi_prior,nu_prior+T+1);
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
stats.arate_noise    = zeros(1,opt.nGibbsIter);
stats.arate_Kf       = zeros(1,opt.nGibbsIter);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Begin Gibbs Sampling Block
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
acc_noise_all = 0; acc_Kf_all = 0; gidx = 1:50; 
for g = 1:opt.nGibbsIter
    % display output
    if mod(g,50) == 0
        arate_noise = acc_noise_all / 50;
        arate_Kf = acc_Kf_all / 50;
        
        disp(['Gibbs iter: ',num2str(g),' arate(noise)=',num2str(arate_noise,'%2.2f'),' arate(Kf)=',num2str(arate_Kf,'%2.2f')]);
        acc_noise_all = 0; 
        acc_Kf_all = 0; 
        
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

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % sample theta
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % whiten f
    %nu = (L_Kt\f);
        
    % sample noise 
    %%%%%%%%%%%%%%
    S2_new = S2 + opt.mh.StepSize*(Noise_proposal*randn(T,1));    

    % make a step in Theta
    Theta_new = [lf; S2_new];
 
    % compute new likelihood
    LogLik_theta_new = gp_mtr(Theta_new, X, Y, opt);
    LogLik_theta_new = -LogLik_theta_new;
    
    % compute new priors for the noise    
    LogPrior_noise_new_all = zeros(length(S2),1);
    for i = 1:length(S2)
         LogPrior_noise_new_all(i) = a_prior*log(b_prior) - gammaln(a_prior) - ...
                                (a_prior+1)*exp(S2_new(i)) - b_prior/exp(S2_new(i));                
    end   
    LogPrior_noise_new = sum(LogPrior_noise_new_all);
    LogPrior_theta_new = LogPrior_Kf + LogPrior_noise_new;
    
    % Accept / reject for Theta
    Ratio = LogLik_theta_new + LogPrior_Kf + LogPrior_theta_new - (LogLik_theta + LogPrior_Kf + LogPrior_theta);
    if Ratio > 0 || (Ratio > log(rand)) % accept
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
    try
        Lf_new = chol(Kf_new)';
        Singular_Kf = false;
    catch
        Lf_new = Lf;
        Singular_Kf = true;
        disp('singular Kf')
    end
    lf_new = Lf_new(CholMask ~= 0);      % extract lower diagonal
    
    Theta_new = [lf_new; S2];
    
    % compute new likelihood
    LogLik_theta_new = gp_mtr(Theta_new, X, Y, opt);
    LogLik_theta_new = -LogLik_theta_new;
    
    % compute new priors for Kf
    [~, LogPrior_Kf_new] = invwishpdf(Kf_new,Psi_prior,nu_prior+T+1);

    % Final priors for theta
    LogPrior_theta_new = LogPrior_Kf_new + LogPrior_noise;
        
    %HastingsRatio = 0.5*(2*w_proposal+3*T+3)*log(det(Kf)) + ...
    %                0.5*w_proposal*(trace(Kf_new\Kf) -trace(Kf\Kf_new)) - ...
    %                0.5*(2*w_proposal+3*T+3)*log(det(Kf_new));         
    HastingsRatio = 0.5*(2*w_proposal+3*T+3)*log(det(Kf)) + ...
                    0.5*w_proposal*(trace(Kf_new/Kf) -trace(Kf/Kf_new)) - ...
                    0.5*(2*w_proposal+3*T+3)*log(det(Kf_new));
  
    Ratio = LogLik_theta_new + LogPrior_theta_new + HastingsRatio - (LogLik_theta + LogPrior_theta);
    if ~Singular_Kf && (Ratio > 0 || (Ratio > log(rand))) % accept
        Theta          = Theta_new;
        lf             = lf_new;
        Kf             = Kf_new;
        LogPrior_Kf    = LogPrior_Kf_new;      
        LogPrior_theta = LogPrior_theta_new;
        LogLik_theta   = LogLik_theta_new;
        
        % update Metropolis proposal distribution
        Psi_proposal   = w_proposal*Kf;
        
        acc_Kf = 1;
    else % reject
        acc_Kf = 0;
    end
    
    Theta_all(:,g) = Theta;
    
    acc_noise_all = acc_noise_all + acc_noise;
    acc_Kf_all = acc_Kf_all + acc_Kf;
    
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

