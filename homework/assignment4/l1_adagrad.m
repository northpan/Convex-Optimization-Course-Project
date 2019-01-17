function [x, out] = l1_adagrad(x0, A, b, mu, opts)
[m, n] = size(A);
pseudomu = 100000*mu;
eps = opts(1);
delta = opts(2);
maxIter = opts(3);
tol = 1e-8;

x = x0;
r = zeros(n, 1);
Atb = A' * b;
AtA = A' * A;

while pseudomu >= mu
    for iter = 1:maxIter
        x0 = x;
        % g is sub gradient
        g = AtA*x - Atb + pseudomu * sign(x); 
        
        r = r + g .* g;

        x = x -  g./(delta + sqrt(r)) * eps;
        
        if norm(x0-x) < tol
            break;
        end
    end
    pseudomu = pseudomu / 10;  
end
out = 0.5 * sum_square(A*x-b) + mu * norm(x, 1);
end
