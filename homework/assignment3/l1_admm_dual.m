function [x, out] = l1_admm_dual(x0, A, b, mu, opts)
[m, n] = size(A);
pseudomu = 10000*mu;
beta = opts(1);
gamma = opts(2);
maxIter = opts(3);

inver = inv(beta*(A*A') + eye(m));

x = zeros(n, 1);
z = zeros(n, 1);

%continuation trick
while pseudomu >= mu
    iter = 0;
    while iter < maxIter                
        y = inver * (beta*A*z + b - A*x);
        z = phi(A'*y + x/beta, pseudomu);
        x = x + gamma*(A'*y-z); 

        iter = iter + 1;
    end
    pseudomu = pseudomu / 10;
end
out = 0.5 * sum_square(A*x-b) + mu * norm(x, 1);
end

function ret = phi(input, mu)
input = min(input, mu);
input = max(input, -mu);
ret = input;
end