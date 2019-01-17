function [x, out] = l1_proximal_grad(x0, A, b, mu, opts)
% use smooth() to replace 1-norm
% smooth(x) = x^2/(2*lambda) if |x|<lambda
%           = |x|-lambda/2   if |x|>=lambda
[n, ~] = size(x0);
AtA = A' * A;
Atb = A' * b;
pseudomu = 10000*mu;
alpha0 = 1/max((eig(AtA))); 
maxIter = 425;
x = x0;

proximal = @(x, t) sign(x).*max(abs(x) - t, 0);

%continuation trick
while pseudomu >= mu
    alpha = alpha0;
    iter = 1;
    while iter < maxIter
        grad = AtA * x - Atb;
        x = x - alpha * grad;
        x = proximal(x, alpha*pseudomu);
        
        iter = iter + 1;
    end
    pseudomu = pseudomu / 10;
end
out = 0.5 * sum_square(A*x-b) + mu * norm(x, 1);
end