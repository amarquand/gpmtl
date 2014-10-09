function Yhat = krr_mtl_laplace(X,Y)

[n,d] = size(Y);

X = (X - repmat(mean(X),n,1)) ./ repmat(std(X),n,1);
Phi = X*X';

% wide
r1       = -5:0.5:8;  % full gridsearch
%r1       = -10;       % set lam(1) = mu = zero
r2       = 0:0.5:8;
% narrow
%r1       = -5:0.1:8;
%r2       = 4:0.1:7;
[l1, l2] = meshgrid(10.^r1, 10.^r2);
lam_all  = [l1(:) l2(:)];
%lam_all = 10.^[4.5 5.5];                 % approx optimal values

RHO  = zeros(d,size(lam_all,1));
MSE  = zeros(d,size(lam_all,1));
SMSE = zeros(d,size(lam_all,1));
for r = 1:size(lam_all,1)
    lam = lam_all(r,:);
    
    Yhat = zeros(n,d);
    for s = 1:n
        tr = [];
        for stest = 1:n
            if stest == s, te = s; else tr = [tr, stest]; end
        end
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
        
        % regulariser - ordinary krr
        %Lam = (lam*n)*eye(size(K));
        
        % regulariser - mtl krr (Baldassarre)
        %Lam = kron(ones(d),diag(omega*ones(ntr,1)));
        %Lam = Lam .* (ones(ntr*d) - eye(ntr*d)) + diag((1-omega)*ones(ntr*d,1));
        
        % regulariser - Laplacian
        L   = diag((d-1)*ones(d,1)) - (ones(d) - eye(d));  % L = D - A
        L   = kron(L,eye(ntr));
        Lam = lam(1)*L + lam(2)*eye(ntr*d);
        
        alpha     = (K+Lam)\(Ytr(:) - Mtr(:));
        Yhat(s,:) = (Ks*alpha)' + mtr;
    end
    RHO(:,r)    = diag(corr(Y,Yhat));
    MSE(:,r)    = (mean((Y-Yhat).^2))';
    SMSE(:,r)   = MSE(:,r) ./ var(Y)';
end
%save([root,'results/results'],'RHO','MSE','SMSE')

% if size(lam_all,1) > 1
%     for scale = 1:size(Y,2)
%         scales{scale}
%         l1a = log10(l1(:)); l2a = log10(l2(:)); 
%         
%         figure
%         subplot(2,1,1)
%         smse = reshape(SMSE(scale,:),length(r2),length(r1));
%         [m,i] = min(SMSE(scale,:));
%         SMSEm = m
%         if length(r1) > 1
%             contourf(r1,r2,smse,25); hold on; colorbar;
%             xlabel('log10(lam1)');
%             ylabel('log10(lam2)');
%             caxis([0.7 1.1])
%             plot(l1a(i),l2a(i),'xk','linewidth',2);
%         else
%             plot(r2,smse); hold on;
%             xlabel('log10(lam2)');
%             plot(l2a(i),m,'xk','linewidth',2); 
%         end
%         title(['Gp ',mat2str(unique(group(id))),' ',scales{scale}, ' - SMSE']);
%         
%         rho = reshape(RHO(scale,:),length(r2),length(r1));
%         [m,i] = max(RHO(scale,:));
%         RHOm = m
%         subplot(2,1,2)
%         if length(r1) > 1
%             contourf(r1,r2,rho,25); hold on; colorbar;
%             xlabel('log10(lam1)');
%             ylabel('log10(lam2)');
%             caxis([0 0.5])
%             plot(l1a(i),l2a(i),'xk','linewidth',2);
%         else
%             plot(r2,rho); hold on;
%             xlabel('log10(lam2)');
%             plot(l2a(i),m,'xk','linewidth',2);
%         end
%         title(['Gp ',mat2str(unique(group(id))),' ',scales{scale}, ' - RHO']);
% 
%     end   
% else
%     for scale = 1:size(Y,2)
%         figure
%         subplot(1,2,1)
%         plot(Yhat(:,scale),Y(:,scale),'x','Markersize',10,'Linewidth',2)
%         xlabel('predicted')
%         ylabel('true')
%         title([scales{scale},' rho = ',num2str(RHO(scale),'%0.2f')])
%         
%         subplot(1,2,2)
%         plot(Y(:,scale));hold on;
%         plot(Yhat(:,scale),'r')
%         legend('true','predicted');
%         title([scales{scale},' SMSE = ',num2str(SMSE(scale),'%0.2f')])
%     end
% end