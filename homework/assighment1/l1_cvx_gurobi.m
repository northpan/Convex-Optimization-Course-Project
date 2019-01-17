function [x, cvx_optval]=l1_cvx_gurobi(x0, A, b, mu, opts2)
[n, ~] = size(x0);
cvx_solver gurobi
cvx_begin 
    variable x(n)
    minimize (mu*norm(x,1) + 0.5*norm(A*x-b))    
cvx_end
end