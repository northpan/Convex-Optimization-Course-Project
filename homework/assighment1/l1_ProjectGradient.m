function [x, cvx_optval] = l1_ProjectGradient(x0, A, b, mu, opts5)
x0 = [max(x0,0); max(-x0,0)];
[~,n] = size(A);
pseodumu = mu * 1e5;
tol = 1e-4;
alpha_l = 1e-6; 
alpha_u = 1;    
alpha = 1e-4;     
iter = 200;
Atb = A' * b;
AtA = A' * A;
while pseodumu >= mu
    cur = pseodumu * ones(2*n,1) - [Atb; -Atb];
    AtAx0 = AtA* (x0(1:n)-x0((n+1):2*n));
    g0 = [AtAx0;-AtAx0] + cur;
    x = x0;
    g = g0;
    k = 1;
    while k < iter
        x0 = x;
        x = max(x - alpha*g, 0);    
        g0 = g;
        AAz = AtA* (x(1:n)-x((n+1):2*n));
        g = [AAz;-AAz] + cur;
        y = g - g0;
        s = x - x0;
        BB = (s'*s) / (s'*y); 
        if s'*y <=0
            alpha = alpha_u;
        else
            alpha = max(alpha_l, min(BB, alpha_u));
        end
        k = k + 1;
        if  norm(max(x-g, 0) - x) < tol 
            break
        end
    end
    pseodumu = pseodumu / 10; 
end
x =  x(1:n)-x((n+1):2*n);
cvx_optval = 0.5*sum_square(A*x - b) + mu*norm(x,1);
end
