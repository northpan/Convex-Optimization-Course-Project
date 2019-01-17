% function Test_l1_regularized_problems

% min 0.5 ||Ax-b||_2^2 + mu*||x||_1

% set seed
% rng(1)

% generate data
n = 1024;
m = 512;

rng('default');
A = randn(m,n);
u = sprandn(n,1,0.1);
b = A*u;
mu = 1e-3;
x0 = rand(n,1);

errfun = @(x1, x2) norm(x1-x2)/(1+norm(x1));

%cvx calling mosek
opts1 = []; %modify options
% 7.2844399533e-02
tic;  
[x1, out1] = l1_cvx_mosek(x0, A, b, mu, opts1);
t1 = toc;

%Augmented Lagrangian method for the dual problem
opts2 = [6.18e-3, 0.00618, 20, 3]; % [beta, gamma, maxIter1, maxIter2]
tic; 
[x2, out2] = l1_auglagrange_dual(x0, A, b, mu, opts2);
t2 = toc;

% Alternating direction method of multipliers for the dual problem
opts3 = [1e-2, 6.18e-3, 20]; % [beta, gamma, maxIter]
tic; 
[x3, out3] = l1_admm_dual(x0, A, b, mu, opts3);
t3 = toc;

% Alternating direction method of multipliers with linearization for the primal problem
opts4 = [30, 1, 600, 0.0003]; 
% [beta, gamma, maxIter, c]
tic; 
[x4, out4] = l1_admm_primal_linear(x0, A, b, mu, opts4);
t4 = toc;

% print comparison results with cvx-call-mosek
fprintf('    cvx_call_mosek: cpu: %5.2f, optval: %3.10e, err-to-cvx-mosek: %3.2e\n', t1, out1, errfun(x1, x1));
fprintf('  auglagrange_dual: cpu: %5.2f, optval: %3.10e, err-to-cvx-mosek: %3.2e\n', t2, out2, errfun(x1, x2));
fprintf('         admm_dual: cpu: %5.2f, optval: %3.10e, err-to-cvx-mosek: %3.2e\n', t3, out3, errfun(x1, x3));
fprintf('admm_primal_linear: cpu: %5.2f, optval: %3.10e, err-to-cvx-mosek: %3.2e\n', t4, out4, errfun(x1, x4));

