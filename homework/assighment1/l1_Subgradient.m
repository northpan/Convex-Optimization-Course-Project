function [x, cvx_optval]=l1_Subgradient(x0, A, b, mu, opts5)
pseudomu = 100; % 10000*mu 
iter = 500; 
tol = 1e-7;
x = x0;

% continuation trick
while pseudomu >= mu
    k = 1;
    alpha = 3e-4;
    while k < iter
        x0 = x;
        g = A' * (A * x - b)  + pseudomu * sign(x);
        if mod(k,50)==0
            alpha = alpha * 0.9;
        end
        x = x - alpha * g;
        if norm(x0-x) < tol
            break;
        end
        k = k + 1;
    end
    pseudomu = pseudomu / 10;
end
cvx_optval = 0.5*sum_square(A*x - b) + mu * norm(x,1);
end