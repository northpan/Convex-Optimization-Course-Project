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

%Gradient method for the smoothed primal problem
opts2 = [0.4*1e-6, 4*1e-4]; % =[lambda, alpha0]
tic; 
[x2, out2] = l1_grad_smooth(x0, A, b, mu, opts2);
t2 = toc;
 
% Fast gradient method for the smoothed primal problem
opts3 = [0.4*1e-6]; % =[lambda]
tic; 
[x3, out3] = l1_fast_grad_smooth(x0, A, b, mu, opts3);
t3 = toc;

% Proximal gradient method for the primal problem
opts4 = []; %modify options
% maxIter=425, tol = 1e-6
tic; 
[x4, out4] = l1_proximal_grad(x0, A, b, mu, opts4);
t4 = toc;

% Fast proximal gradient method for the primal problem
opts5 = []; %modify options
tic; 
[x5, out5] = l1_fast_proximal_grad(x0, A, b, mu, opts5);
t5 = toc;

% print comparison results with cvx-call-mosek
fprintf('  cvx_call_mosek: cpu: %5.2f, optval: %3.10e, err-to-cvx-mosek: %3.2e\n', t1, out1, errfun(x1, x1));
fprintf('     grad_smooth: cpu: %5.2f, optval: %3.10e, err-to-cvx-mosek: %3.2e\n', t2, out2, errfun(x1, x2));
fprintf('fast_grad_smooth: cpu: %5.2f, optval: %3.10e, err-to-cvx-mosek: %3.2e\n', t3, out3, errfun(x1, x3));
fprintf('       prox_grad: cpu: %5.2f, optval: %3.10e, err-to-cvx-mosek: %3.2e\n', t4, out4, errfun(x1, x4));
fprintf('  fast_prox_grad: cpu: %5.2f, optval: %3.10e, err-to-cvx-mosek: %3.2e\n', t5, out5, errfun(x1, x5));
