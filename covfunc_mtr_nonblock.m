function [K, dK] = covfunc_mtr_nonblock(in, LogTheta, B)    

% Computes covariance by forming a product between the task covariance 
% (i.e. task kernel) with the input covariance (i.e. base kernel) according
% to:
%       k(f_p(x),g_q(z)) = Kf(p,q) * Kx(x,z)
%
% This function takes the inputs: {Kx,M}, Logtheta and B (='diag' or the 
% index of the hyperparameter which we are taking the derivative of). The
% matrix M is a one of t coding matrix indicating which task each sample
% belongs to.
% The hyperparamters are:
% 
% Logtheta = [ lf(:)
%              log(sn(:)) ]
% 
% where lf = tril(Lf) is the lower diagonal of the cholesky decomposition
% of the T x T task covariance and sf2 is a T x 1 vector collecting the 
% noise variance for each of the tasks.

Kx    = in{1};
M     = in{2};

%N     = size(Kx,1);
T     = size(M,2);
lmaxi = (T*(T+1)/2);

% Reconstruct chol(Kx)' and Kf
Lf     = zeros(T);
lf     = LogTheta(1:lmaxi);
id     = tril(true(T));
Lf(id) = lf;
Kf     = Lf*Lf';

% output noise
sf2 = exp(2 * LogTheta(lmaxi+1:end)); % sf2 = exp(Logtheta(...)^2)

K = (M*Kf*M').*Kx + diag(M*sf2);
K = K + 1e-5*eye(size(K,1));
if nargout > 1
    if B <= lmaxi
        ID     = zeros(T);
        ID(id) = (1:lmaxi)';
        J      = double(ID == B);
        %[p,q]  = find(ID == B);
        %J      = zeros(T);
        %J(p,q) = 1;
        
        %dK = kron(J*Lf',Kx) + kron(Lf*J',Kx);
        dK = M*(J*Lf'+Lf*J')*M'.*Kx;
    else
        p      = B - lmaxi;
        %J      = zeros(T);
        %J(p,p) = 1;
        
        %dK = kron(2*sf2(p)*J,eye(N));
        
        j = zeros(T,1);
        j(p) = 1;
        
        dK = diag(2*sf2(p)*M*j);
    end
else
    if nargin > 3 % are there test data?
        if strcmp(B,'diag') && numel(B)>0;   % determine mode
            K = diag(K);
        else
            error ('covfunc_mtr: unknown mode');
        end
    end
end