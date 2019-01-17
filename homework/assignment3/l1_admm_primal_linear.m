function [x, out] = l1_admm_primal_linear(x0, A, b, mu, opts)
[m, n] = size(A);
pseudomu = 10000*mu;
beta = opts(1);
gamma = opts(2);
maxIter1 = opts(3);
c = opts(4);

x = x0;
u = zeros(n, 1);
z = zeros(n, 1);
Atb = A' * b;
AtA = A' * A;
Q = inv(AtA + beta*eye(n));

thresh = @(x,th) sign(x).* max(abs(x) - th,0);

while pseudomu >= mu
    for iter = 1:maxIter1
        x = x - c*(AtA*x + beta*x + beta*(u-z) - Atb);        
        z = thresh(x + u, pseudomu/beta);
        u = u + gamma * (x - z); 
    end
    pseudomu = pseudomu / 10;  
end
out = 0.5 * sum_square(A*x-b) + mu * norm(x, 1);
end
