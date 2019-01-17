function [x, out] = l1_fast_proximal_grad(x0, A, b, mu, opts)
AtA = A' * A;
Atb = A' * b;
pseudomu = 10000*mu;
alpha0 = 1/max((eig(AtA))); 
maxIter = 100;
x = x0;

proximal = @(x, t) sign(x).*max(abs(x) - t, 0);

%continuation trick
while pseudomu >= mu
    alpha = alpha0;
    iter = 0;
    v = x;
    while iter < maxIter
        x_old = x;
        theta = 2/(iter + 1);
        y = (1 - theta)*x + theta*v;
        grad = AtA * y - Atb;
        x = proximal(y - alpha * grad, alpha*pseudomu);
        v = x + (1/theta)*(x - x_old);

        iter = iter + 1;
    end
    pseudomu = pseudomu / 10;
end
out = 0.5 * sum_square(A*x-b) + mu * norm(x, 1);
end