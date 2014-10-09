function [f, s2] = gp_pred_mtr(C,Cs,css,Y)

if iscell(Y)
    y  = Y{1};
else
    y  = Y(:);   
end


%C   = feval(opt.CovFunc,Kx,LogTheta);
%Cs  = feval(opt.CovFunc,Kxs,LogTheta);
%css = feval(opt.CovFunc,Kxs,LogTheta,'diag');

Lc = chol(C)';                        % cholesky factorization of the covariance
alpha = solve_chol(Lc',y);

f  = Cs*alpha;
v  = Lc\(Cs');
s2 = css - v'*v;
