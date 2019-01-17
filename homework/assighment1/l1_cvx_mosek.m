function [x, cvx_optval]=l1_cvx_mosek(x0, A, b, mu, opts1)
[n, ~] = size(x0); 
cvx_solver mosek
cvx_begin 
    variable x(n);
    minimize (mu*norm(x,1) + 0.5*norm(A*x-b))    
cvx_end
end