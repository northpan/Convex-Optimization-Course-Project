function [x, out] = l1_rmsprop(x0, A, b, mu, opts)
[m, n] = size(A);
pseudomu = 100000*mu;
eps = opts(1);
rho = opts(2);
delta = opts(3);
maxIter = opts(4);
tol = 1e-10;

x = x0;
r = zeros(n, 1);
v = zeros(n, 1);
Atb = A' * b;
AtA = A' * A;

i = 1;
while pseudomu >= mu
    for iter = 1:maxIter                
        g = AtA*x - Atb + pseudomu * sign(x); 
        r = rho*r + (1-rho)* (g.*g);
        x = x - (eps ./ (sqrt(r) + delta)).*g;
    end
    pseudomu = pseudomu / 10; 
    i = i + 1;
end
out = 0.5 * sum_square(A*x-b) + mu * norm(x, 1);
end
