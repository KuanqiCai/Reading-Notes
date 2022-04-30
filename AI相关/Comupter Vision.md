# 一、Wissenswertes über Bilder

## 1. Darstellung von Bildern

- Image Representation

  1. **Continuouse Representation** (Kontinuierliche Darstellung)

     As a function of two variables(to derive algorithms)
     $$
     I:R^2\supset\Omega\rightarrow R,\ (x,y)\rightarrow I(x,y)
     $$

     - $I$ is differentiable(differenzierbar)
     - $\Omega$ is simply connected and bounded

  2. **Discrete Representation**(Diskrete Darstellung)

      as a Matrix $I\in R^{m\times n}$ the Item $I_{k,l}$ corresponds to the Intensity value强度。

     The scale is typically between [0,255] or [0,1]

- Discrete Sampling

  1. Sampling of a one-dimensional signal
     $$
     S\{f(x)\}=(...,f(x-1),f(x),f(x+1),...)
     $$
  
  2. Sampling of an image
     $$
     S\{I(x,y)\}=
      \left[
      \begin{matrix}
        \ddots & \vdots &\vdots &\vdots & \\
        \cdots & I(x-1,y-1) & I(x-1,y) & I(x-1,y+1)&\cdots \\
        \cdots & I(x,y-1) & I(x,y) & I(x,y+1)&\cdots \\
        \cdots & I(x+1,y-1) & I(x+1,y) & I(x+1,y+1)&\cdots \\
         & \vdots &\vdots &\vdots & \ddots \\
       \end{matrix}
       \right]
     $$
  
     - Assumption: Origin is in the top-left corner
  
     

## 2. Bildgradient

image gradient 是一个用于determine确定local intensity changes的重要工具

- Edges(Kanten)边界

  - Edges correspond to stark local changes of the intensity.

  - Local changes are described by a gradient
    $$
    \nabla I(x,y)= \left[
     \begin{matrix}
       \frac{d}{dx}I(x,y) \\
       \frac{d}{dy}I(x,y)
      \end{matrix}
      \right]
    $$

    - $I \in R^{m\times n}$ is known
    - Naive approach to estimate the gradient
      - $\frac{d}{dx}I(x,y)\approx I(x+1,y)-I(x,y)$
      - $\frac{d}{dy}I(x,y)\approx I(x,y+1)-I(x,y)$

- Interpolation内插法

  从discrete signal $f[x]=S\{f(x)\}$到 continuous signal $f(x)$

  使用Sample Value(Abtastwerte) 和 Interpolationsfilter 的Faltung卷积来求得：
  $$
  f(x) \approx \sum_{k=-\infty}^\infty f[k]h(x-k)=: f[x]*h(x)
  $$

  - Interpolation filter
    1. Gaussian filter: $h(x)=g(x)=Ce^{\frac{-x^2}{2\sigma^2}}$
    2. Ideales Interpolationsfilter: $h(x)=sinc(x)=\frac{sin(\pi x)}{\pi x},sinc(0)=1$

- Discrete Derivative

  Discrete Derivative离散导数是通过对interpolated signal内插信号求导计算而得到的：

  算法步骤Algorithmically:

  1. Reconstruction重构of the continuous Signal
  2. Differentiation of the continuous signal
  3. Sampling of the Derivative导数

  Derivation:
  $$
  \begin{aligned}
  f^{\prime}(x)&\approx\frac{d}{dx}(f[x]*h(x))\\
  &=f[x]*h^{\prime}(x) \\
  \\
  f^{\prime}[x]&=f[x]*h^{\prime}[x]\\
  &=\sum_kf[x-k]h^{\prime}[k]
  \end{aligned}
  $$

  - 2D-Rekonstruktion
    $$
    I(x,y)\approx I[x,y]*h(x,y)=\sum_{k=-\infty}^{\infty}\sum_{l=-\infty}^\infty I[k,l]g(x-k)g(y-l)\\
    h(x,y):=g(x)g(y)\  \text{这里用separable 2D Gaussian filter}
    $$

    - 在实际中会用一数n来代替无穷的求和。

    - 高斯过滤器的C 的选择：

      Normalization(Normierung)正则化C，使得所有的gewichte权重相加趋向1
      $$
      C=\frac{1}{\sum_{k=-n}^ne^{\frac{-k^2}{2\sigma^2}}}
      $$

  - 2-D Derivative

    利用高斯过滤的Separability可分性来计算梯度：

    Ableitung in X-Richtung
    $$
    \begin{aligned}
    \frac{d}{dx}I(x,y)&\approx I[x,y]*(\frac{d}{dx}h(x,y))\\
    &=\sum_{k,l}I[k,l]g^{\prime}(x-k)g(y-l)\\
    S\{\frac{d}{dx}I(x,y)\}&=I[x,y]*g^{\prime}[x]*g[y]\\
    &=\sum_{k,l}I[x-k,y-l]g^{\prime}[k]g[l]
    \end{aligned}
    $$

  - Sobel-Filter索伯算子

    Sobel Filter是integer整数approximations of the double gradient.

    - Approxiamtion von $S\{\frac{d}{dx}I(x,y)\}=I[x,y]*g^{\prime}[x]*g[y]=\sum_{k=-\infty}^\infty \sum_{l=-\infty}^\infty I[x-k,y-l]g^{\prime}[k]g[l]$

      durch endliche Summe$\sum_{k=-1,0,1}\sum_{l=-1,0,1}I[x-k,y-l]g^{\prime}[k]g[l]$

    - Normierungsfaktor$C=\frac{1}{1+2e^{-\frac{1}{2\sigma^2}}}$

      

## 3.Merkmalspunkte-Ecken und Kanten

特征点Feature points – 角Corners and 边缘Edges

- Harris Corner and Edge Detector
  - Corner: shifts移动 in all directions cause a change
  - Edge: shifts in all directions except exactly one cause a change
  - Homogenous surface均匀的平面: no change, independently of the direction不管在哪个方向都没变化  

- Formal Definition of change

  - Position in the image: $x=\left[\begin{matrix}x_1\\x_2\end{matrix}\right]$，$I(x)=I(x_1,x_2)$

  - Shift direction $u=\left[\begin{matrix}u_1\\u_2\end{matrix}\right]$

    - Shift移动 according to vector u

  - Change in the image segment段

    $S(u)=\displaystyle\int_W(I(x+u)-I(x))^2dx$

  - Differentiability可微性 of $I$:
    $$
    \displaystyle\lim_{u\rightarrow0}\frac{I(x+u)-I(x)-\nabla I(x)^Tu}{||u||}=0
    $$

    - Approximation for small shifts: $I(x+u)-I(x)\approx\nabla I(x)^Tu$

  - Approximation of the change in the image segment
    $$
    S(u)=\displaystyle\int_W(I(x+u)-I(x))^2dx\approx\displaystyle\int_W(\nabla I(x)^Tu)^2dx=u^T(\displaystyle\int_W\nabla I(x)\nabla I(x)^Tdx)u
    $$

  - 由上式可得Harris matrix: $G(x)=\displaystyle\int_W\nabla I(x)\nabla I(x)^Tdx$
    $$
    \nabla I(x)\nabla I(x)^T=
     \left[
     \begin{matrix}
       (\frac{\partial}{\partial x_1}I(x))^2 & \frac{\partial}{\partial x_1}I(x)\frac{\partial}{\partial x_2}I(x) \\
       \frac{\partial}{\partial x_2}I(x)\frac{\partial}{\partial x_1}I(x) & (\frac{\partial}{\partial x_2}I(x))^2  \\
      \end{matrix}
      \right]
    $$
    由此可改写Image segment为：$S(u)\approx u^TG(x)u$

- Excursus附记: Linear Algebra

  - Real valued symmetrical Matrix对称矩阵：$A=A^T$

    - positive definite正定的：$x^TAx>0,x\neq0$
    - positive semi-definite 半正定的：$x^TAx\geq0$

  - Some Examples

    - Zero-Matrix is positive semi-definite but not positive definite
    - All positive definite matrices are also positive semi-definite.
    - The identity matrix单位矩阵 is positive definite
    - Harris matrix is positive semi-definite

  - Eigenvalue decomposition分解 of real-valued symmetrical Matrices

    All real-valued symmetrical matrices $A=A^T$ can be decomposed分解 into a product $A=V\Lambda V^T$ with $VV^T=I$ and a diagonal matrix对角矩阵$\Lambda$, wherein the eigenvalues of $A$ lie on the diagonal. The columns of $V$ are the corresponding eigenvectors.

  - 如果$\lambda_1...\lambda_n$是矩阵G的特征值，那么：

    - $det\ G=\displaystyle\prod_{i=1}^{n}\lambda_i$ (determinant行列式  is the product of the eigenvalues)
    - $tr\ G=\displaystyle\sum_{i=1}^n\lambda_i$(The trace迹(spur) is the sum of the eigenvalues )

- Eigenvalue decomposition

  - Eigenvalue decomposition分解 of Harris matrix
    $$
    G(x)=\displaystyle\int_W\nabla I(x)\nabla I(x)^Tdx=V \left[
     \begin{matrix}
       \lambda_1 &  \\
        & \lambda_2  \\
      \end{matrix}
      \right]V^T\\
      with\ VV^T=I_2 \ and\ eigenvalues\ \lambda_1\geq\lambda_2\geq0
    $$

  - Change is dependent on the eigenvectors: $V=[v_1,v_2]$
    $$
    S(u)\approx u^TG(x)u=\lambda_1(u^Tv_1)^2+\lambda_2(u^Tv_2)^2
    $$
    有3种情况：

    1. Both eigenvalues positive
       - $S(u)>0$ for all $u$ (change in every direction)
       - Examinated image section被研究的图片部分 contains a corner
    2. One eigenvalue positive, one eigenvalue equal to zero
       - $S(u)\begin{cases}=0,\ falls\ u=rv_2\ \text{(no change, only in the direction of the eigenvector with eigenvalue 0)}\\>0,\ sonst\end{cases}$
       - Examinated image section contains an edge
    3. Both eigenvalues equal to 0
       - $S(u)=0$for all u (no change in any direction)
       - Examinated image section contains a homogenous surface

    但在现实中由于噪声、离散取样或者数值上的不准确，导致Eigenvalues永远不可能是0所以：

    1. Corner: two large eigenvalue
    2. Edge: a large eigenvalue and a small eigenvalue
    3. Homogeneous surface: two small eigenvalues

- Practical Implementation of the Harris detector

  - Computation of the Harris matrix

    - Approximate G(x) with a finite sum
      $$
      G(x)=\displaystyle\int_W\nabla I(x)\nabla I(x)^Tdx\approx \displaystyle\sum_{\tilde{x}\in W(x)}\nabla I(\tilde{x})\nabla I(\tilde{x})^T
      $$

    - Weighted Sum dependent on the position of $\tilde{x}$
      $$
      G(x)\approx\displaystyle\sum_{\tilde{x}\in W(x)}w(\tilde{x})\nabla I(\tilde{x})\nabla I(\tilde{x})^T
      $$

    - Weights $w(\tilde{x})>0$ emphasize the influence of the central pixel

  - A simple criterium标准 for corners and edges

    - Analyze the quantity$H:=det(G)-k(tr(G))^2$

      $H=(1-2k)\lambda_1\lambda_2-k(\lambda_1^2+\lambda_2^2)$

    - Corner: H larger than a positive threshold value.$0<\tau_+<H$

    - Edge: H smaller than a negative threshold value.$H<\tau_-<0$

    - Homogeneous surface: H small. $\tau_-<H<\tau_+$

## 4. Korrespondenzschätzung für Merkmalspunkte

Correspondence Estimation of Feature Points特征点的对应估计

- 问题描述

  - Two Image $I_1:\Omega_1\rightarrow R,I_2:\Omega_2\rightarrow R$ from the same 3D scene are known
  - Find pairs of image points$(x^{(i)},y^{(i)})\in \Omega_1\times\Omega_2$ which correspond to the same 3D points
  - Feature points$\{x_1,...x_n\}\subset\Omega_1$and $\{y_1,...y_n\}\subset\Omega_2$ are known

- Naive Solution to the Problem

  **方法1、Sum of squared differences(SSD)**

  - Examine image sections $V_i$ around $x_i$ and $W_i$ around $y_i$ in matrix representation and compare the respective intensity values.

  - Formal Definition

    - A criterion判断标准：$d(V,W)=||V-W||^2_F$

      - 这里$||A||^2_F=\displaystyle\sum_{kl}A^2_{kl}=trace(A*A)$描述了quadratic二次的 Frobenius norm范数

    - 找一个对于$V_j$合适的$W_j$来使$j=arg\ \displaystyle\min_{k=1,..,n}d(V_i,W_k)$. 反过来找V也OK.

  - SSD方法的缺点：
  
    Change in Illumination光亮 or Rotation。为了克服它需要Normalization正则化亮度和旋转
  
    - **Rotation Normalization**: By means of the Gradient Direction
  
      Pre-processing:
  
      1. Determine the gradient in all feature points.
      2. Rotate the regions around feature points旋转2个图片中的1个 such that the gradient points in one direction.
      3. Extrapolate推断 V,W from the rotated regions.
  
    - Bias and Gain Model:
  
      - $\alpha$：Scaling缩放 of the intensity values(Gain)。
  
        $\beta$：Shift 移动 of the intensity (Bias)。
  
        Gain Model: $W\approx\alpha V$
  
        Bias Model: $W\approx V+\beta 1 1^T$
  
        - 这里$1=(1,...,1)^T$
  
        **Bias and Gain Model**: $W\approx \alpha V+\beta11^T$
      
      1. Calculate the intensity mean平均值
      
      $$
      \begin{aligned}
      \bar{W}&=\frac{1}{N}(11^TW11^T)\\
      &\approx \frac{1}{N}(11^T(\alpha V+\beta11^T)11^T)\\
      &=\alpha\frac{1}{N}(11^TV11^T)+\beta11^T\\
      &=\alpha\bar{V}+\beta11^T
      \end{aligned}
      $$
      
      2. Subtract the mean-matrix
      
      $$
      \begin{aligned}
      W-\bar{W}&\approx \alpha V+\beta 11^T-(\alpha\bar{V}+\beta11^T)\\
      &=\alpha(V-\bar{V})
      \end{aligned}
      $$
      
      3. Standard Deviation标准差 of the intensity
      
      $$
      \begin{aligned}
      \sigma(W)&=\sqrt{\frac{1}{N-1}||W-\bar{W}||^2_F}\\
      &=\sqrt{\frac{1}{N-1}tr((W-\bar{W})^T(W-\bar{W}))}\\
      &\approx \sqrt{\frac{1}{N-1}tr(\alpha(V-\bar{V})^T\alpha(V-\bar{V}))}\\
      &=\alpha\sigma(V)
      \end{aligned}
      $$
      
      4. Normalization正则化 of the image sections
         $$
         \begin{aligned}
         W_n:&=\frac{1}{\sigma(W)}(W-\bar{W})\\
         &\approx \frac{1}{\alpha\sigma(V)}(\alpha(V-\bar{V}))\\
         &=\frac{1}{\sigma(V)}(V-\bar{V})\\
         &=:V_n
         \end{aligned}
         $$
    
    **方法2、Normalized Cross Correlation(NCC)**归一化互相关
    
    Derivation起源于SSD
    
    - SSD of two normalized image sections:$d(V,W)=||V_N-W_N||^2_F=2(N-1)-2tr(W_n^TV_n)$
    
    - The Normalized Cross Correlation of th two image sections is defined as $NCC=\frac{1}{N-1}tr(W_n^TV_n)$
    - $-1\leq NCC\leq1$
    - 两个normalized image section是相似的,如果
      1. SSD small (few differences)
      2. NCC close to =1 (high correlation)
