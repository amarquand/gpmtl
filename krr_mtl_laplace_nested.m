function [Yhat, lam, results] = krr_mtl_laplace_nested(X,Y,trid,teid)

[n,d] = size(Y);
nte   = length(teid);
ntr   = length(trid);

% wide
r1       = -5:0.5:8;  % full gridsearch
%r1       = -5;       % set lam(1) = mu = zero or inf
r2       = 0:0.5:8;
% narrow
%r1       = -5:0.1:8;
%r2       = 4:0.1:7;
[l1, l2] = meshgrid(10.^r1, 10.^r2);
lam_all  = [l1(:) l2(:)]; 
l1ae = l1(:); l2ae = l2(:);

rm = trid;
te = teid;
n_folds = 4;

fprintf('Standardising features ...\n')
Xz = (X - repmat(mean(X(rm,:)),n,1)) ./ repmat(std(X(rm,:)),n,1);
Phi = Xz*Xz';

%fprintf('Normalizing kernel ...\n')
%Phi  = prt_normalise_kernel(Phi);

RHO_va  = zeros(d,size(lam_all,1));
MSE_va  = zeros(d,size(lam_all,1));
SMSE_va = zeros(d,size(lam_all,1));
fprintf('Running nested cross-validation ...\n')
for r = 1:size(lam_all,1)
    lam = lam_all(r,:);
    % begin nested loop
    Yhat_va = zeros(ntr,d);
    for f2 = 1:n_folds
        srange = 1:floor(ntr/n_folds);
        tr = [];
        for ftest = 1:n_folds
            if ftest == f2
                va = rm(srange); vamatid = srange; 
            else
                tr = [tr rm(srange)];
            end
            
            if ftest == n_folds-1
                srange = max(srange+1):length(rm);
            else
                srange = srange + floor(ntr/n_folds);
            end
        end
        tr = sort(tr); va = sort(va); nva = length(va);
        
        K = {}; Ks = {};
        for k = 1:d
            K{k}   = Phi(tr,tr);
            Ks{k}  = Phi(va,tr);
        end
        K   = blkdiag(K{:});
        Ks  = blkdiag(Ks{:});
        Ytr = Y(tr,:);
        
        % training mean
        mtr = mean(Y(tr,:));
        Mtr = repmat(mtr,ntr-nva,1);
        
        % regulariser
        Lam  = regulariser('graph',lam,ntr-nva,d);
        
        alpha              = (K+Lam)\(Ytr(:) - Mtr(:));
        Yhat_va(vamatid,:) = reshape(Ks*alpha,nva,d) + repmat(mtr,nva,1);
    end
    
    RHO_va(:,r)    = diag(corr(Y(rm,:),Yhat_va));
    MSE_va(:,r)    = (mean((Y(rm,:)-Yhat_va).^2))';
    SMSE_va(:,r)   = MSE_va(:,r) ./ var(Y(rm,:))';
end
fprintf('Making predictions ...\n')
% begin testing block
%figure
%for task = 1:d
    [m,i] = min(mean(SMSE_va));
    %[m,i] = min(SMSE_va(task,:));
    
    % make predictions
    lam = [l1ae(i) l2ae(i)];
    %Param(s,:) = lam;
    
    tr  = rm;
    ntr = length(tr);
    
    K = {}; Ks = {};
    for k = 1:d
        K{k}   = Phi(tr,tr);
        Ks{k}  = Phi(te,tr);
    end
    K   = blkdiag(K{:});
    Ks  = blkdiag(Ks{:});
    Ytr = Y(tr,:);
    
    % training mean
    mtr = mean(Y(tr,:));
    Mtr = repmat(mtr,ntr,1);
    
    % regulariser
    Lam  = regulariser('graph',lam,ntr,d);
    
    alpha     = (K+Lam)\(Ytr(:) - Mtr(:));
    %%Yhat(s,:) = (Ks*alpha)' + mtr;
    %YhatT = (Ks*alpha)' + mtr;
    %Yhat(:,task) = YhatT(:,task);
    
    Yhat = reshape(Ks*alpha,nte,d) + repmat(mtr,nte,1);
    
%end

RHO    = diag(corr(Y(te,:),Yhat));
MSE    = (mean((Y(te,:)-Yhat).^2))';
SMSE   = MSE ./ var(Y)';
%save([root,'results/results_all'],'Y','Yhat','RHO','MSE','SMSE')
results = [SMSE RHO];