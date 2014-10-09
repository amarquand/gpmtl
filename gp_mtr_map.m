function [pennlml, dpennlml] = gp_mtr_map(LogTheta, K,Y, opt)

[N,T] = size(Y);

%a_prior = 1;
%b_prior = 1;   
% Ym = Y - repmat(mean(Y),N,1);
% YY = Ym'*Ym;
% Psi_prior = scale_prior*YY;

% a_prior = 3;
% b_prior = 0.5;
% Psi_prior = opt.Psi_prior;
% nu_prior = T*5;
a_prior = 1;
b_prior = 3;
Psi_prior = opt.Psi_prior;
nu_prior = T+2;

lmaxi = (T*(T+1)/2);

% Reconstruct chol(Kx)' and Kf
Lf     = zeros(T);
lf     = LogTheta(1:lmaxi);
id     = tril(true(T));
Lf(id) = lf;
Kf     = Lf*Lf';
Kf     = Kf + 0.1*eye(T);

% output noise
%sn = exp(2 * LogTheta(lmaxi+1:end));
sn = exp(LogTheta(lmaxi+1:end));

if nargout == 1
    nlml = gp_mtr(LogTheta, K,Y, opt);
else
%    try
        [nlml dnlml] =  gp_mtr(LogTheta, K,Y, opt);
%     catch
%         nlml = nan;
%         dnlml = nan(size(LogTheta));
%     end
end

% compute priors for 
lP_noise  = zeros(size(sn));
dlP_noise = zeros(size(sn));
for i = 1:length(sn)
    % inverse Gamma
    lP_noise(i)  = a_prior*log(b_prior) - gammaln(a_prior) - ...
                   (a_prior+1)*sn(i) - b_prior/sn(i);  
    dlP_noise(i) = -(a_prior+1) + b_prior / sn(i);        
end

% compute priors for Kf
[~, lP_Kf] = invwishpdf(Kf,Psi_prior,2*nu_prior+1);

% derivatives
dlP_Kf = zeros(size(lf));
Lf_iKf_Psi_iKf = Lf'*Kf\Psi_prior/Kf;
for i = 1:length(lf);
    ID     = zeros(T);
    ID(id) = (1:lmaxi)';
    J      = double(ID == i);
    
    dlP_Kf(i) = -0.5*(2*nu_prior+1)*trace(Kf\(Lf*J+J*Lf')) - 0.5*trace(-2*Lf_iKf_Psi_iKf*J);
end

pennlml  = nlml - lP_Kf - sum(lP_noise);
if nargout > 1
    dpennlml = dnlml - [dlP_Kf; dlP_noise];
end
