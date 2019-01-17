function [x, out] = l1_momentum(x0, A, b, mu, opts)
[m, n] = size(A);
pseudomu = 100000*mu;
eps = opts(1);
alpha = opts(2);
maxIter = opts(3);

x = x0;
v = zeros(n, 1);
Atb = A' * b;
AtA = A' * A;

i = 1;
while pseudomu >= mu
    for iter = 1:maxIter       
        g = AtA*x - Atb + pseudomu * sign(x); 
        v = alpha*v - eps*g;
        x = x + v;
    end
    pseudomu = pseudomu / 10; 
    i = i + 1;
end
out = 0.5 * sum_square(A*x-b) + mu * norm(x, 1);
end
