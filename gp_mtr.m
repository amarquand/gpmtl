function [nlml, dnlml, alpha] = gp_mtr(LogTheta, K,Y, opt)

if iscell(Y)
    y  = Y{1};
    NT = length(y);
else
    [N,T] = size(Y);
    
    y  = Y(:);
    NT = N*T;    
end
%Kx = X*X';

C = feval(opt.CovFunc,K,LogTheta);
%C = C + 0.01*eye(size(C));

Lc = chol(C)';                        % cholesky factorization of the covariance
alpha = solve_chol(Lc',y);

nlml = 0.5*y'*alpha + sum(log(diag(Lc))) + 0.5*NT*log(2*pi);
 
if nargout > 1
    dnlml = zeros(size(LogTheta));       % set the size of the derivative vector
    W = Lc'\(Lc\eye(NT))-alpha*alpha';                % precompute for convenience
    for i = 1:length(dnlml)
        [~,dC] = feval(opt.CovFunc,K,LogTheta,i);
        dnlml(i) = sum(sum(W.*dC))/2; % 0.5*tr(W*dK)
    end
end