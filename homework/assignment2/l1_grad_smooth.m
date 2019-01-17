function [x, out] = l1_grad_smooth(x0, A, b, mu, opts)
% use smooth() to replace 1-norm
% smooth(x) = x^2/(2*lambda) if |x|<lambda
%           = |x|-lambda/2   if |x|>=lambda
[n, ~] = size(x0);
AtA = A' * A;
Atb = A' * b;
pseudomu = 10000*mu;
lambda = opts(1);
alpha0 = 1/max((eig(AtA))); % 3.3888e-04
maxIter = 700;
tol = 1e-6;

x = x0;
%continuation trick
while pseudomu >= mu
    alpha = alpha0; %
    iter = 1;
    grad = ones(n,1);
    while norm(grad) > tol && iter < maxIter
        grad = AtA * x - Atb + pseudomu * grad_smooth(x, lambda);
        x = x - alpha * grad;
        if norm(x0 - x) < tol
            break
        end
        iter = iter + 1;
    end
    pseudomu = pseudomu / 10;
end
out = 0.5 * sum_square(A*x-b) + mu * norm(x, 1);
end

function ret = grad_smooth(x, lambda)
mask = abs(x) < lambda;
x(mask) = x(mask) / lambda;
x(~mask) = sign(x(~mask));
ret = x;
end