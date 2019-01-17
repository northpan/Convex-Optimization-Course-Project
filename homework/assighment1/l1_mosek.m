function [x, cvx_optval]=l1_mosek(x0, A, b, mu, opts3)
[~,n] = size(A);
q = [0.5*(A'*A), zeros(n,n);zeros(n,n),zeros(n,n)];
c = [zeros(n,1);mu*ones(n,1)];
a = [eye(n),eye(n);-eye(n),eye(n)];
v = A\b;
blc = [-v;v];
res = mskqpopt(q,c,a,blc,[],[],[],[],'minimize');
x = res.sol.itr.xx(1:1024)+v;
cvx_optval = res.sol.itr.pobjval;
end