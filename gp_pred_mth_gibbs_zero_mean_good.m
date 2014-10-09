function [Pred, Noise, Kf] = gp_pred_mth_gibbs(X, tr, te, Y, Vtr, Vte, opt)

% Subject and cross-validation parameters
[~,T] = size(Y);
Ntr   = length(tr);
Nte   = length(te);
N     = Ntr + Nte;

% id matrices to keep track of everything.
V     = X{3};  
Rid   = X{4};
lmaxi = (T*(T+1)/2);
id    = tril(true(T));

% What follows is a complicated way to figure out the indices belonging to 
% the training and test data. The final indices can be used to entries in 
% the kernel matrices and labels
trv = []; tev = []; v0 = 0;
for t = 1:T, trv = [trv tr+v0]; tev = [tev te+v0]; v0 = v0 + N; end
trl = zeros(N*T,1); trl(trv) = 1;
tel = zeros(N*T,1); tel(tev) = 1;
trk = logical(V*trl);
tek = logical(V*tel);

% These indices are used to keep track of which samples in the training and
% test sets are regression tasks and which are classification tasks
trrid = Rid(tr,:);   trrid = logical(Vtr*trrid(:));
terid = Rid(te,:);   terid = logical(Vte*terid(:));
trcid = 1-Rid(tr,:); trcid = logical(Vtr*trcid(:));
tecid = 1-Rid(te,:); tecid = logical(Vte*tecid(:));

% Load the posteriors
disp('++ Loading posteriors ...');
load([opt.OutputFilename,'f_all'])
load([opt.OutputFilename,'Theta_all'])
load([opt.OutputFilename,'alpha_all'])
Theta_post = Theta_all(:,opt.BurnIn:opt.TestInterval:end);
alpha_post = alpha_all(:,opt.BurnIn:opt.TestInterval:end);
f_post     = f_all(:,opt.BurnIn:opt.TestInterval:end);

% Compute predictions
disp('++ Computing predictions ...');
Pred           = zeros(sum(tecid)+sum(terid), 1);
Noise          = zeros(sum(Rid(1,:)),1); 
Kf             = zeros(T,T);
n_test_samples = 0;
mus_all = []; mu_all = []; Pred_all = []; Kf_all = zeros(T,T,size(f_post,2));
for i = 1 : length(Theta_post);
    hyp   = Theta_post(:,i);
    f     = f_post(:,i);
    alpha = alpha_post(:,i);
    
    C   = feval(opt.CovFunc,X,hyp);
    K   = C(trk,trk);
    Ks  = C(tek,trk);
    %kss = C(tek,tek);
    
    %%%%%%%%%%%%%%%%%%
    % Regression tasks
    %%%%%%%%%%%%%%%%%%
    fr_i        = Ks(terid,trrid)*alpha(trrid);   
    Pred(terid) = Pred(terid) + fr_i;
    
    %%%%%%%%%%%%%%%%%%%%%%
    % Classification tasks
    %%%%%%%%%%%%%%%%%%%%%%  
    %InvKt = inv(K);
    if opt.nTestSamples > 1
        error('Binary classification. Only one test sample required');
    else
        %CondPrior_muc = K(trcid,trrid)/K(trrid,trrid)*f(trrid);
        CondPrior_Kc  = K(trcid,trcid) - K(trcid,trrid)/K(trrid,trrid)*K(trcid,trrid)';
        
        fc = f(trcid);% - CondPrior_muc; %mean has already been subtracted
        
        %trid = [find(trrid); find(trcid)];
        %mus  = K(tecid,trid)/K(trid,trid)*f(trid);
        %mus  = K(tecid,trrid)/K(trrid,trrid)*f(trrid);
        mus  = K(tecid,trcid)/CondPrior_Kc*fc;    % this is the correct way
    
        mu   = Ks(tecid,trcid)/CondPrior_Kc*fc + mus;
        P_i  = likelihood_multinomial(mu,[f(tecid) -f(tecid)]);
        
        % debugging
        %mus_all = [mus_all mus];
        %mu_all =  [mu_all mu];
    end
    Pred(tecid) = Pred(tecid) + P_i(:,1);
  
    n_test_samples = n_test_samples + 1;
    
    % Reconstruct chol(Kx)' and Kf
    Lf     = zeros(T);
    lf     = hyp(1:lmaxi)';
    Lf(id) = lf;
    Kf     = Kf + Lf*Lf';
    %Noise  = Noise + exp(hyp(end-T+1:end));
    Noise  = Noise + hyp(lmaxi+1:end);
    
    %Pred_all =  [Pred_all Pred];
    Kf_all(:,:,i) = Lf*Lf';
end
Pred  = Pred / n_test_samples;
Kf    = Kf / n_test_samples;
Noise = Noise / n_test_samples;