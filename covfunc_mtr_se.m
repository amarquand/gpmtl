function [K, dK] = covfunc_mtr_se(in, LogTheta, B)    

% Computes covariance by forming a product between the task covariance 
% (i.e. task kernel) with the input covariance (i.e. base kernel) according
% to:
%       k(f_p(x),g_q(z)) = Kf(p,q) * Kx(x,z)
%
% This function takes the inputs: {Kx,Y}, Logtheta and B (='diag' or the 
% index of the hyperparameter which we are taking the derivative of).
%
% The task kernel (Kf) is a covariance matrix.
%
% The input kernel (Kx) is a squared Exponential covariance function with 
% isotropic distance measure. The covariance function is parameterized as:
%
% k(x^p,x^q) = sf^2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2) 
%
% where the P matrix is ell^2 times the unit matrix and sf^2 is the signal
% variance. The hyperparamters are:
% 
% Logtheta = [ lf(:)
%              log(ell)
%              log(sf)
%              log(sn2(:)) ]
% 
% where lf = tril(Lf) is the lower diagonal of the cholesky decomposition
% of the T x T task covariance and sn2 is a T x 1 vector collecting the 
% noise variance for each of the tasks.

Kx    = in{1};
Y     = in{2};

[N,T] = size(Y);
lmaxi = (T*(T+1)/2);

% Reconstruct chol(Kx)' and Kf
Lf     = zeros(T);
lf     = LogTheta(1:lmaxi);
id     = tril(true(T));
Lf(id) = lf;
Kf     = Lf*Lf';

% parameters of squared exponential kernel
ell = exp(LogTheta(lmaxi+1));
sf2 = exp(2 * LogTheta(lmaxi+2));

% output noise
sn2 = exp(2 * LogTheta(lmaxi+3:end)); % sn2 = exp(Logtheta(...)^2)

% compute SE kernel
Kxl = Kx/(ell^2);
%Kx  = 2*sf2*exp(-ell^2)*exp(Kxl);
Kx  = sf2*exp(-ell^(-2))*exp(Kxl);

K = kron(Kf,Kx) + kron(diag(sn2),eye(N));
%K = K + 1e-5*eye(size(K,1));
if nargout > 1
    if B <= lmaxi % dK / d(Lpq)
        ID     = zeros(T);
        ID(id) = (1:lmaxi)';
        J      = double(ID == B);
        
        dK = kron(J*Lf',Kx) + kron(Lf*J',Kx);
    else % derivatives
        if B == lmaxi+1 % dK / d(ell)
            dKx = 2*sf2*exp(-ell^2)*exp(Kxl).*(1-2*Kxl/ell)/(ell^2);
            %dKx = sf2*exp(-Kxl/2).*Kxl;
            
            dK = kron(Kf,dKx);
        elseif B == lmaxi+2 % dK / d(sf2)
            dKx = 2*sf2 * exp(-ell^2)*exp(Kxl);
            %dKx = sf2*exp(-Kxl/2);
            
            dK = kron(Kf,dKx);
        else % dK / d(sn2)
            p      = B - (lmaxi+2);
            J      = zeros(T);
            J(p,p) = 1;
            
            dK = kron(2*sn2(p)*J,eye(N));
        end
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