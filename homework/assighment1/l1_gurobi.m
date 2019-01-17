function [x, cvx_optval]=l1_gurobi(x0, A, b, mu, opts4)
[~,n] = size(A);
At = [A -A];
model.Q = 0.5*sparse(At'*At);

c = mu*ones(2*n,1)- (b'*At)';
model.obj =c;
P =eye(2*n);
model.A = sparse(P);
l1 = zeros(2*n,1);
model.rhs =full(l1);
model.sense = '>';
params.method = 2;

res = gurobi(model, params);

x = res.x(1:n,1)-res.x(n+1:2*n,1);
cvx_optval = 0.5 * sum_square(A * x - b) + mu * norm(x, 1);
end