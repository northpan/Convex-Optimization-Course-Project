直接调用mosek（不通过cvx）需要先在Matlab里加一句：

addpath /Users/narsilzhang/Downloads/mosek/8/toolbox/r2014a/



Continuation

$\arg\min_x \frac{1}{2}\Vert Ax-b\Vert^2_2+\mu\Vert x\Vert_1 $

对于大的$\mu$，该问题收敛快

因此可以选取一列$\mu_1\ge\mu_2\ge\cdots\ge\mu$，每次使用上一个优化问题的结果作为初始值



