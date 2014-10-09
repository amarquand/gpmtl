function [K, dK] = covfunc_mth(in, LogTheta, B)    

% Computes covariance by forming a product between the task covariance 
% (i.e. task kernel) with the input covariance (i.e. base kernel) according
% to:
%       k(f_p(x),g_q(z)) = Kf(p,q) * Kx(x,z)
%
% This function takes the inputs: {Kx,M,V,R}, Logtheta and B ('diag' or the 
% index of the hyperparameter which we are taking the derivative of). The
% matrix M is a one of t coding matrix indicating which task each sample
% belongs to. V is a matrix indicating which data are missing and R
% indicates which of the different outputs are regression tasks.
% The hyperparamters are:
% 
% Logtheta = [ lf(:) ]
% 
% where lf = tril(Lf) is the lower diagonal of the cholesky decomposition
% of the T x T task covariance. Note that this function will not add noise
% to the covariance for regression tasks.

Kx    = in{1};  % covariance between all data points
M     = in{2};  % task memberships
V     = in{3};  % missing data
%R     = in{4};  % which tasks are regression

% Note that the following assumes a multi-output model and needs to be
% modified to accommodate the general case
N     = size(Kx,1);
T     = size(M,2);
lmaxi = (T*(T+1)/2);

% deal with missing data
%Kx = V*kron(ones(T),Kx)*V';
Kx = V*kkron(ones(T),Kx)*V';  %faster
Mb = zeros(N*T,T);
for t = 1:T
    Mb(N*(t-1)+(1:N),t) = M(:,t);%.*R(:,t);
end    
M  = V*Mb;

% Reconstruct chol(Kx)' and Kf
Lf     = zeros(T);
lf     = LogTheta(1:lmaxi);
id     = tril(true(T));
Lf(id) = lf;
Kf     = Lf*Lf';

% % test to compensate for imbalance
% m  = sqrt(sum(sum(M)) ./ sum(M)');
% m  = m ./ min(m);
% Kf = Kf .* (m*m');

% output noise
%sn2 = exp(2 * LogTheta(lmaxi+1:end)); % sn2 = exp(Logtheta(...)^2)

% compute kernel
%K = (M*Kf*M').*Kx + diag(M*sn2); 
K = (M*Kf*M').*Kx; 

% add a ridge for stability
K = K + 1e-5*eye(size(K,1));

% % derivatives
% if nargout > 1
%     if B <= lmaxi
%         ID     = zeros(T);
%         ID(id) = (1:lmaxi)';
%         J      = double(ID == B);
%         dK     = M*(J*Lf'+Lf*J')*M'.*Kx;
%         
%         %[p,q]  = find(ID == B);
%         %J      = zeros(T);
%         %J(p,q) = 1;
%         %dK     = kron(J*Lf',Kx) + kron(Lf*J',Kx);
%     else
%         p      = B - lmaxi;
%         j      = zeros(T,1);
%         j(p)   = 1;
%         dK     = diag(2*sn2(p)*M*j);
%         
%         %J      = zeros(T);
%         %J(p,p) = 1;
%         %dK     = kron(2*sn2(p)*J,eye(N));  
%     end
% else
%     if nargin > 3 % are there test data?
%         if strcmp(B,'diag') && numel(B)>0;   % determine mode
%             K = diag(K);
%         else
%             error ('covfunc_mtr: unknown mode');
%         end
%     end
% end