function [LogPost, dLogPost] = gp_mth_hmc_posterior_f(fc, Tc, InvCc, LogDetCc, muc, InvCr, LogDetCr, fr, A)

if nargin < 5
    muc = 0;
end

mur = A*fc;

if nargout == 1
    [tmp, LogLik] = likelihood_multinomial(fc, Tc);
    LogPrior      = prior_log_gauss(fc,InvCc,LogDetCc,muc) + ...
                    prior_log_gauss(fr,InvCr,LogDetCr,mur);
else
    [tmp, LogLik, dLogLik]  = likelihood_multinomial(fc, Tc);
    [LogPriorc, dLogPriorc] = prior_log_gauss(fc,InvCc,LogDetCc,muc);
    
    B = A'*InvCr*A;
    dLogPriorr = (1-0.5*(B+B'))*fc;
    
    LogPrior = LogPriorc + prior_log_gauss(fr,InvCr,LogDetCr,mur);
    dLogPost = -(dLogLik + dLogPriorc + dLogPriorr);
end

LogPost = -(LogLik + LogPrior);
        
        