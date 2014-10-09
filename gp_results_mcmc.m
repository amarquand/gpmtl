function [] = results(plot_variables)

try
    plot_variables;
catch
    plot_variables = false;
end

[statsname, working_dir] = uigetfile('','Select MCMC stats file');
[tname, tdir] = uigetfile('','Select chains for MH variable');

test_interval = 50;%100;

load([working_dir,statsname]);

% Exclude burn-in samples
load([tdir,tname]);
    
srange     = stats.opt.BurnIn:test_interval:stats.opt.nGibbsIter;
Theta_post = Theta_all(:,srange);

theta0 = zeros(size(Theta_post,1),1);

ESS = results_ESS((Theta_post)',length(Theta_post)-1);

N = length(Theta_post);
disp(['mean ESS (theta) = ',num2str(100*mean(ESS)/N),'%']);
disp(['min ESS (theta)  = ',num2str(100*min(ESS)/N),'%']);

stats
if plot_variables
    results_generate_plots((Theta_post),theta0);
end

[tmp, tmin] = min(ESS);
[tmp, tmax] = max(ESS);

m = median(ESS);
[tmp, tmed] = min(abs(ESS - m));

ti = [tmin, tmed, tmax];

 figure
tthin = Theta_post(:,1:test_interval:end);
%subplot(2,1,1)
plot(tthin(ti,1:min(length(tthin),1000))');
xlabel('iteration')
ylabel('log(theta)')
title('Trace: theta')
legend('min','med','max');
%subplot(2,1,2)
%plot_acf_pretty(tthin(tmin,1:min(length(tthin),1000))')

end

