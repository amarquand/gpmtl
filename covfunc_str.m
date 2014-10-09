function [K, dK] = covfunc_str(in, LogTheta, B)    

% Computes covariance by forming a product between the task covariance 
% (i.e. task kernel) with the input covariance (i.e. base kernel) according
% to:
%       k(f_p(x),g_q(z)) = Kf(p,q) * Kx(x,z)
%
% This function takes the inputs: {Kx,Y}, Logtheta and B (='diag' or the 
% index of the hyperparameter which we are taking the derivative of).
% The hyperparamters are:
% 
% Logtheta = [ log(sf2(:)) ]
% 
% where lf = tril(Lf) is the lower diagonal of the cholesky decomposition
% of the T x T task covariance and sf2 is a T x 1 vector collecting the 
% noise variance for each of the tasks.

Kx    = in{1};
Y     = in{2};

[N,T] = size(Y);

% Reconstruct chol(Kx)' and Kf
Kf     = eye(T);

% output noise
sf2 = exp(2 * LogTheta); % sf2 = exp(Logtheta(...)^2)

K = kron(Kf,Kx) + kron(diag(sf2),eye(N));
K = K + 1e-5*eye(size(K,1));
if nargout > 1
    p      = B;
    J      = zeros(T);
    J(p,p) = 1;
    
    dK = kron(2*sf2(p)*J,eye(N));
    
else
    if nargin > 3 % are there test data?
        if strcmp(B,'diag') && numel(B)>0;   % determine mode
            K = diag(K);
        else
            error ('covfunc_mtr: unknown mode');
        end
    end
end