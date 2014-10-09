function [stats] = gpr_mrf_gibbs_mh(X, Y, opt)

% Subject and cross-validation parameters
if iscell(Y) % non-block design
    N = length(Y{1});
    T = size(X{2},2);
    
    Ys = X{2};
    y  = Y{1};
else
    [N,T] = size(Y);  
end

RandStream.setGlobalStream(RandStream('mt19937ar','seed',sum(100*clock)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MCMC Parameter Specification
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generic MCMC parameters
write_interval = round(opt.nGibbsIter/10);

% initialise top-level priors
Q_prior = opt.PriorParam(1);  % MRF precision matrix
a_prior = opt.PriorParam(2);  % Parameters of prior for the noise
b_prior = opt.PriorParam(3); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% starting likelihood for theta
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Theta        = repmat(opt.X0_Theta,1,size(Y,2));
%LogLik_theta = gp_mtr(Theta, X, Y, opt);
opt.type2ml = false;
%opt.hyp0    = Theta;
%[~, LogLik_theta] = gp_mcmc_torque(Theta,X,Y,opt);
LogLik_theta = torque_batch(Theta,X,Y,opt,5000);
LogLik_theta = -LogLik_theta;

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% starting priors for theta
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% theta = [log(ell), log(sf), log(sn)] 
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Private function block
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [LML] = torque_batch(Theta,X,Y,opt,Nperjob,Xs)
try
    Nperjob;
catch
    Nperjob = 1000; % suitable if optimising hyperparameters
end
Njobs = ceil(size(Y,2) / Nperjob);
timreq = 100;

fprintf('Partitioning data matrices into jobs ...\n');
Yb     = cell(Njobs,1);
Xb     = cell(Njobs,1);
Xsb    = cell(Njobs,1);
Thetab = cell(Njobs,1);
optb   = cell(Njobs,1);

Yhat = zeros(size(Y));
S2   = zeros(size(Y));
%Z    = zeros(size(Y));
LML = ones(size(Y,2),1);
%HYP  = ones(size(Y,2),length(unwrap(opt.hyp0)),Nfold);
for b = 1:Njobs
    if b < Njobs
        id = (1:Nperjob)+(b-1)*Nperjob;
    else
        id = ((b-1)*Nperjob+1):size(Y,2);
    end
    
    Thetab{b}   = Theta(:,id);
    Xb{b}       = X;
    Yb{b}       = Y(:,id); %Yb{b}  = Yt(tr,:);
    optb{b}     = opt;
    if nargout > 2
        Xsb{b}      = Xs;
    end
end

%run the jobs
try
    if nargout > 2
        [nlmls,yhat,s2] = qsubcellfun('gp_mcmc_torque',Thetab,Xb,Yb,Xsb,optb,'memreq', 50*1024^2, 'timreq', timreq);
    else
        [nlmls] = qsubcellfun('gp_mcmc_torque',Thetab,Xb,Yb,Xsb,optb,'memreq', 50*1024^2, 'timreq', timreq);
    end
catch
    disp('torque failed! Trying again ...');
    try
        if nargout > 2
            [hyp,nlmls,yhat,s2] = qsubcellfun('gp_mcmc_torque',Xb,Yb,Xsb,optb,'memreq', 50*1024^2, 'timreq', timreq);
        else
            [hyp,nlmls] = qsubcellfun('gp_mcmc_torque',Xb,Yb,Xsb,optb,'memreq', 50*1024^2, 'timreq', timreq);
        end
    catch
        error('torque failed twice. Aborting.');
    end
end

% reassemble output
fprintf('Reassembling job output into matrices ...\n');
for b = 1:Njobs
    if b < Njobs
        id = (1:Nperjob)+(b-1)*Nperjob;
    else
        id = ((b-1)*Nperjob+1):size(Y,2);
    end
    LML(id)  = -nlmls{b};
    
    if nargout > 2
        Yhat(:,id) = yhat{b};
        S2(:,id)   = s2{b};
    end
end
end
