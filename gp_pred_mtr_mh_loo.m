function [Fs, Noise, Kf] = gp_pred_mtr_mh(X, tr, te, Ytr, opt)

% Subject and cross-validation parameters
[~,T] = size(Ytr);
Ntr   = length(tr);
Nte   = length(te);
N     = Ntr + Nte;

% Load the posteriors
disp('++ Loading posteriors ...');
load([opt.OutputFilename,'Theta_all'])

Theta_post = Theta_all(:,opt.BurnIn:opt.TestInterval:end);

% Compute predictions
disp('++ Computing predictions ...');
Fs = zeros(Nte, T);
n_test_samples = 0;
lmaxi = (T*(T+1)/2);
id    = tril(true(T));
Noise = zeros(Nte,T); 
Kf    = zeros(T,T);
for i = 1 : length(Theta_post);
    hyp = Theta_post(:,i);
    
    C = feval(opt.CovFunc,X,hyp);
    ktr = []; kte = []; offset = 0;
    for k = 1:T
       ktr = [ktr tr+offset];
       kte = [kte te+offset];
       offset = offset + N;
    end
    K   = C(ktr,ktr);
    Ks  = C(kte,ktr);
    kss = C(kte,kte);
    
    Fs_i = gp_pred_mtr(K,Ks,kss,Ytr)';
    Fs   = Fs + Fs_i;
    
    n_test_samples = n_test_samples + 1;
    
    % Reconstruct chol(Kx)' and Kf
    Lf     = zeros(T);
    lf     = hyp(1:lmaxi)';
    Lf(id) = lf;
    Kf     = Kf + Lf*Lf';
    Noise  = Noise + exp(hyp(lmaxi+1:end))';    
end

Fs    = Fs / n_test_samples;
Kf    = Kf / n_test_samples;
Noise = Noise / n_test_samples;