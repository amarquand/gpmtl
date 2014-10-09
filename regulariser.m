function Lam = regulariser(type, lam, n, d)

switch type
    case 'krr'
        % regulariser - ordinary krr
        Lam = (lam*n)*eye(n*d);
    case 'mtl'
        % regulariser - mtl krr (Baldassarre)
        Lam = kron(ones(d),diag(omega*ones(n,1)));
        Lam = Lam .* (ones(n*d) - eye(n*d)) + diag((1-omega)*ones(n*d,1));   
    case 'graph'
        % regulariser - Laplacian
        L   = diag((d-1)*ones(d,1)) - (ones(d) - eye(d));  % L = D - A
        L   = kron(L,eye(n));
        Lam = lam(1)*L + lam(2)*eye(n*d);    
    otherwise
        error(['Unknown regulariser type: ',type]);     
end

