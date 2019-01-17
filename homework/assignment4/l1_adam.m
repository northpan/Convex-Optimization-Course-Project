function [x, out] = l1_adam(x0, A, b, mu, opts)
[m, n] = size(A);
pseudomu = 100000*mu;
eps = opts(1);
delta = opts(2);
rho1 = opts(3);
rho2 = opts(4);
maxIter = opts(5);
tol = 1e-8;

x = x0;
r = zeros(n, 1);
s = zeros(n, 1);
Atb = A' * b;
AtA = A' * A;

while pseudomu >= mu
    for iter = 1:maxIter
        x0 = x;
        % g is sub gradient 
        g = AtA*x - Atb + pseudomu * sign(x); 

        s = rho1 * s + (1-rho1) * g;
        r = rho2 * r + (1-rho2) * (g.*g);
        
        s_hat = s / (1 - rho1^iter);
        r_hat = r / (1 - rho2^iter);
        
        h = s_hat ./ (delta + sqrt(r_hat));
        x = x - eps * h;
        
        if norm(x0-x) < tol
            break;
        end
    end
    pseudomu = pseudomu / 10;  
end
out = 0.5 * sum_square(A*x-b) + mu * norm(x, 1);
end
