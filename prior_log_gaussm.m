function [logP dlogP] = prior_log_gauss(f,invC, LogDetC,alpha) 

% old name: RMHMC_prior_f

% if nargin < 4
%     mu = 0;
% end
n = size(invC,1);

%logP = -0.5*(f-mu)'*invC*(f-mu) - 0.5*LogDetC - 0.5*n*log(2*pi);
logP = -0.5*f'*alpha - 0.5*LogDetC - 0.5*n*log(2*pi);
%dlogP = -invC*f;
dlogP = -invC*f;

end