function [W] = gp_weights_mtr_mh(K, X, Y, opt)

% Subject and cross-validation parameters
[Ntr,T] = size(Y);

% Load the posteriors
disp('++ Loading posteriors ...');
load([opt.OutputFilename,'Theta_all'])

Theta_post = Theta_all(:,opt.BurnIn:opt.TestInterval:end);

% Compute predictions
disp('++ Computing predictions ...');
W     = zeros(T,size(X,2));
n_test_samples = 0;
for i = 1 : length(Theta_post);
    if ~mod(i,500)
        fprintf('++++ %d of %d done.\n',i,length(Theta_post));
    end
    hyp = Theta_post(:,i);
    
    [~,~, alphai] = gp_mtr(hyp, K,Y, opt);
    
    Wi =  reshape(alphai,Ntr,T)'*X;
    W = W + Wi;
    
    n_test_samples = n_test_samples + 1;
end

W = W / n_test_samples;