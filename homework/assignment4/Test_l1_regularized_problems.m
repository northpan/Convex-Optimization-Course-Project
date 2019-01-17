% function Test_l1_regularized_problems

% min 0.5 ||Ax-b||_2^2 + mu*||x||_1

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
tic;  
[x1, out1] = l1_cvx_mosek(x0, A, b, mu, opts1);
t1 = toc;

% AdaGrad
opts2 = [0.9, 1e-8, 400]; 
fprintf("%.5f, %.1e, %d\n", opts2(1), opts2(2), opts2(3));
tic; 
[x2, out2] = l1_adagrad(x0, A, b, mu, opts2);
t2 = toc;

% Adam
opts3 = [8e-2, 1e-8, 1-5e-2, 1-1e-3, 150]; 
fprintf("%.5f, %.6e, %.1e, %.1e, %d\n", opts3(1), opts3(2), opts3(3), opts3(4), opts3(5));
tic; 
[x3, out3] = l1_adam(x0, A, b, mu, opts3);
t3 = toc;


% RMSProp
opts4 = [6*1e-4, 1-1e-6, 1e-8, 290]; 
% [eps, rho, delta, maxIter]
fprintf("%.5f, %.1e, %.7f, %d\n", opts4(1), opts4(2), opts4(3), opts4(4));
tic; 
[x4, out4] = l1_rmsprop(x0, A, b, mu, opts4);
t4 = toc;

% Momentum
opts5 = [3e-4, 3.3e-4, 400]; 
% [eps, alpha, maxIter] [8e-5, 0.3*1e-3, 2000]
fprintf("%.5f, %.1e, %d\n", opts5(1), opts5(2), opts5(3));
tic; 
[x5, out5] = l1_momentum(x0, A, b, mu, opts5);
t5 = toc;

% print comparison results with cvx-call-mosek
fprintf('    cvx_call_mosek: cpu: %5.2f, optval: %3.10e, err-to-cvx-mosek: %3.2e\n', t1, out1, errfun(x1, x1));
fprintf('           adagrad: cpu: %5.2f, optval: %3.10e, err-to-cvx-mosek: %3.2e\n', t2, out2, errfun(x1, x2));
fprintf('              adam: cpu: %5.2f, optval: %3.10e, err-to-cvx-mosek: %3.2e\n', t3, out3, errfun(x1, x3));
fprintf('           rmsprop: cpu: %5.2f, optval: %3.10e, err-to-cvx-mosek: %3.2e\n', t4, out4, errfun(x1, x4));
fprintf('          momentum: cpu: %5.2f, optval: %3.10e, err-to-cvx-mosek: %3.2e\n', t5, out5, errfun(x1, x5));

