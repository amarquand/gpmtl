function [LogPost, dLogPost] = hmc_posterior_f(f, T, InvC, LogDetC, mu, alpha)

if nargin < 5
    mu = 0;
end

if nargout == 1
    [tmp, LogLik] = likelihood_multinomial(f, T);
    LogPrior      = prior_log_gaussm(f,InvC,LogDetC,alpha);
else
    [tmp, LogLik, dLogLik] = likelihood_multinomial(f, T);
    [LogPrior, dLogPrior]  = prior_log_gaussm(f,InvC,LogDetC,alpha);
    
    dLogPost = -(dLogLik + dLogPrior);
end

LogPost = -(LogLik + LogPrior);
        
        