# 基本介绍

## CasADi是啥

- CasADi 一开始是一个algorithmic differentiation(AD)算法微分的工具，使用computer algebra systems(CAS)计算机代数系统的语法。

- 现在还加入了 

  1. ordinary differential equations常微分方程(ODE)/differential-algebraic equations微分代数方程(DAE) integration and sensitivity analysis

  2.  nonlinear programming 

  3. interfaces to other numerical tools

- 所以现在它是一个基于梯度gradient-based的数值优化numerical optimization的通用工具，擅长于optimal control最优控制

## Symbolic framework符号框架

- SX symbolics:
  - `SX.sym(name,n,m)`: Create an **n**-by-**m** symbolic primitive初始的
  - `SX.zeros(n,m)`: Create an **n**-by-**m** dense matrix with all zeros
  - `SX(n,m)`: Create an **n**-by-**m** sparse matrix with all **structural** zeros
  - `SX.ones(n,m)`: Create an **n**-by-**m** dense matrix with all ones
  - `SX.eye(n)`: Create an **n**-by-**n** diagonal matrix with ones on the diagonal and structural zeros elsewhere.
  - `SX(scalar_type)`: Create a scalar (1-by-1 matrix) with value given by the argument. This method can be used explicitly明确的, e.g. `SX(9)`, or implicitly暗示的, e.g. `9 * SX.ones(2,2)`.
  - `SX(matrix_type)`: Create a matrix given a numerical matrix given as a NumPy or SciPy matrix (in Python) or as a dense or sparse matrix (in MATLAB/Octave). In MATLAB/Octave e.g. `SX([1,2,3,4])` for a row vector, `SX([1;2;3;4])` for a column vector and `SX([1,2;3,4])` for a 2-by-2 matrix. This method can be used explicitly or implicitly.
  - `repmat(v,n,m)`: Repeat expression **v n** times vertically and **m** times horizontally. `repmat(SX(3),2,1)` will create a 2-by-1 matrix with all elements 3.
  - (*Python only*) `SX(list)`: Create a column vector (n-by-1 matrix) with the elements in the list, e.g. `SX([1,2,3,4])` (note the difference between Python lists and MATLAB/Octave horizontal concatenation, which both uses square bracket syntax)
  - (*Python only*) `SX(list of list)`: Create a dense matrix with the elements in the lists, e.g. `SX([[1,2],[3,4]])` or a row vector (1-by-n matrix) using `SX([[1,2,3,4]])`.

- MX symbolics:

  - 语法和sx一致

  - 但和sx不同，MX的运算不限于标量的unary一元或binary二元运算(R->R or RXR->R)。

    MX允许多稀疏矩阵multiple sparse-matrix valued输入，和多稀疏矩阵输出的函数。