function [x, out] = l1_auglagrange_dual(x0, A, b, mu, opts)
[m, n] = size(A);
pseudomu = 10000*mu;
beta = opts(1);
gamma = opts(2);
maxIter1 = opts(3);
maxIter2 = opts(4);
tol = 1e-6;

x = zeros(n, 1);
y = zeros(m, 1);

%continuation trick
while pseudomu >= mu     
    iter1 = 1;
    while iter1 < maxIter1
        % y sub-problem: gradient descent
        iter2 = 1;
         while iter2 < maxIter2
            temp = softThreshold(A'*y+x/beta, pseudomu);
            grad = y - b + beta*A*temp;
            y = y - 0.1 * grad;

            iter2 = iter2 + 1;
         end
        
        temp=x/beta+A'*y;
        z = phi(temp, pseudomu); 
        
        x0 = x;
        x = x + gamma*(A'*y-z);
        
        if norm(x0-x) < tol
            break;
        end
        
        iter1 = iter1 + 1;
    end
pseudomu = pseudomu / 10;
end
out = 0.5 * sum_square(A*x-b) + mu * norm(x, 1);
end

function [x] = softThreshold(x, mu)
    x = sign(x) .* max(abs(x) - mu, 0);
end

function L =lagrangian(A,b,y,x,beta,mu)
    temp=softThreshold(A'*y+x/beta,mu);
    L=-b'*y+0.5*y'*y+0.5*beta*temp'*temp;
end

function ret = phi(input, mu)
input = min(input, mu);
input = max(input, -mu);
ret = input;
end