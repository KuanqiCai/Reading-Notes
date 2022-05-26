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

  - Sobel-Filter索伯滤波器

    索贝尔算子（Sobeloperator）主要用作边缘检测，在技术上，它是一离散性差分算子，用来运算图像亮度函数的灰度之近似值。在图像的任何一点使用此算子，将会产生对应的灰度矢量或是其法矢量。

    Sobel Filter是integer整数approximations of the double gradient.

    -  该算子由2个3X3矩阵g[x],g[y]组成，分别代表横向和纵向。比如：

      | g[x] |      |      |      | g[y] |      |      |      |
      | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
      |      | -1   | 0    | 1    |      | 1    | 2    | 1    |
      |      | -2   | 0    | 2    |      | 0    | 0    | 0    |
      |      | -1   | 0    | 1    |      | -1   | -2   | -1   |
    
      如果I代表原始图像，那么Gx,Gy就分别代表经横向及纵向边缘检测的图像灰度值
    
      $G_x=g[x]*I$; $G_y=g[y]*I$      这里*是卷积的意思
    
      该点的灰度大小为：$G=\sqrt{G_x^2+G_y^2}$
    
    - Approxiamtion von $S\{\frac{d}{dx}I(x,y)\}=I[x,y]*g^{\prime}[x]*g[y]=\sum_{k=-\infty}^\infty \sum_{l=-\infty}^\infty I[x-k,y-l]g^{\prime}[k]g[l]$
    
      durch endliche Summe$\sum_{k=-1,0,1}\sum_{l=-1,0,1}I[x-k,y-l]g^{\prime}[k]g[l]$
    
    - Normierungsfaktor$C=\frac{1}{1+2e^{-\frac{1}{2\sigma^2}}}$
    
      

## 3.Merkmalspunkte-Ecken und Kanten

特征点Feature points – 角Corners and 边缘Edges

哈里斯角点检测的基本思想：

算法基本思想是使用一个固定窗口在图像上进行任意方向上的滑动，比较滑动前与滑动后两种情况，窗口中的像素灰度变化程度，如果存在任意方向上的滑动，都有着较大灰度变化，那么我们可以认为该窗口中存在角点。

- Harris Corner and Edge Detector
  - Corner: shifts移动 in all directions cause a change
  - Edge: shifts in all directions except exactly one cause a change
  - Homogenous surface均匀的平面: no change, independently of the direction不管在哪个方向都没变化  

- Formal Definition of change

  - Position in the image: $x=\left[\begin{matrix}x_1\\x_2\end{matrix}\right]$，$I(x)=I(x_1,x_2)$

  - Shift direction $u=\left[\begin{matrix}u_1\\u_2\end{matrix}\right]$

    - Shift移动 according to vector u

  - Change in the image segment段：当窗口发生u移动时，那么滑动前与滑动后对应的窗口中的像素点灰度变化描述如下：

    $S(u)=\displaystyle\int_W(I(x+u)-I(x))^2dx$

    也可写成：$S(u_1,u_2)=\displaystyle\sum_{x_1,x_2}w(x_1,x_2)[I(x_1+u_1,x_2+u_2)-I(x_1,x_2)]^2$。这里的W是window function窗口函数，也即我们选中的segment。

    后面3步是在化简该式

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
    由此可改写Image segment为：$S(u)\approx u^TG(x)u=[u_1\ u_2](\sum \left[
     \begin{matrix}
       I_{x_1}^2 & I_{x_1}I_{x_2} \\
       I_{x_1}I_{x_2} & I_{x_2}^2  \\
      \end{matrix}
      \right])\left[\begin{matrix}u_1\\u_2\end{matrix}\right]$

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

- Eigenvalue decomposition：

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

## 4.Korrespondenzschätzung für Merkmalspunkte

Correspondence Estimation of Feature Points特征点的对应估计

- 问题描述

  - Two Image $I_1:\Omega_1\rightarrow R,I_2:\Omega_2\rightarrow R$ from the same 3D scene are known
  - Find pairs of image points$(x^{(i)},y^{(i)})\in \Omega_1\times\Omega_2$ which correspond to the same 3D points
  - Feature points$\{x_1,...x_n\}\subset\Omega_1$and $\{y_1,...y_n\}\subset\Omega_2$ are known

- Naive Solution to the Problem

  **方法1、Sum of squared differences(SSD)**

  - Examine image sections $V_i$ around $x_i$ and $W_i$ around $y_i$ in matrix representation and compare the respective intensity values.

    - 这里$x_i,y_i$是2张图片各自的特征点不是横纵坐标的意思

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
  
    Derivation起源于SSD，同样需要进行上面那4步窗口正则化**参考作业2.1**
  
    - SSD方法：SSD of two normalized image sections:$d(V,W)=||V_N-W_N||^2_F=2(N-1)-2tr(W_n^TV_n)$
      - 这里的**N**是窗口内所有的像素个数：即`window_length*window_length`
  
    - NCC方法：The Normalized Cross Correlation of th two image sections is defined as $NCC=\frac{1}{N-1}tr(W_n^TV_n)$
      - 注意这里和上面公式中$W_n和V_n$的顺序，
    - $-1\leq NCC\leq1$，相关度超出阈值的部分要舍弃(=0)
    - 两个normalized image section是相似的,如果
      1. SSD small (few differences)
      2. NCC close to =1 (high correlation)

# 二、Bildentstehung

## 1.Bildentstehung

Image Formation成像

- The Pinhole Camera Model(Lochkameramodell)

  对于lens透镜：
  $$
  |\frac{b}{B}|=\frac{z-f}{f}\\
  |\frac{z}{Z}|=|\frac{b}{B}|\\
  \frac{1}{|Z|}+\frac{1}{z}=\frac{1}{f}
  $$
  其中：

  - $f$:焦距，焦点到凸透镜中心的距离。
    - 焦点：平行于主光轴的光，经透镜折射后与主光轴相交的点
    - 弥散圆circle of confusion:在焦点前后，光线会扩散为一个圆，成像变得模糊
    - 焦深：离焦点一定范围内的弥散圆还是让人清晰可见的，这段离焦点的距离称之为景深。显然景深包含焦点的左右两侧，靠近透镜那一侧称之为前景深，远离透镜那一侧称之为后景深
    - 景深：焦深对应于像，而景深对应于物体。
  - $z$:像距，成像平面到凸透镜的距离。f<=z<=2f
    - 像距受物距、焦距而影响
  - $Z$:物距，物体到凸透镜的距离
  - $b$:像的大小。$B$:物的大小

- Projection of a point in Space onto the Focal Plane成像平面

  - 小孔成像实际就是将**相机坐标系**中的三维点变换到成像平面中的**图形坐标系**
  
    - 相机坐标系(三维)
  
      相机的中心被称为焦点或者广信，以焦点O为原点和坐标X,Y,Z组成了相机坐标系。
  
    - 图像坐标系(二维)
  
      成像平面focal plane中，以成像平面的中心O'为原点和坐标轴x',y'组成了图像坐标系
  
      - 注意不是像平面，虽然实际中光线经过透镜后并不会完美的都交于焦点，而会形成弥散圆。
      - [相机成像究竟是成在像平面还是成在焦平面？底片相当于像平面还是焦平面？](https://www.zhihu.com/question/33793912/answer/57646234)
  
  - 如果空间点P在相机坐标系中的坐标是$P_c=[X,Y,Z]^T$ ，其像点在图像坐标系中的坐标是$p=[x,y]^T$,因为光轴和成像平面垂直，所以像点p在相机坐标系中的坐标是$p=[x,y,z]^T$，z=f(相机焦距)
  
    根据三角形相似关系可得：
    $$
    \begin{cases}
    x=f\frac{X}{Z}\\
    y=f\frac{Y}{Z}\\
    z=f
    \end{cases}\tag{1}
    $$
    可以记作从3d->2d坐标转换时,坐标都乘了$\frac{f}{Z}$

## 2.Homogene Koordinaten

Homogeneous Coordinates齐次坐标

上面的1式描述了3D空间到2D平面的映射，但该映射对于坐标Z来说是非线性的(因为Z是分母！=0)。所以为了方便的同意处理X,Y,Z三个坐标轴的数据，就需要引入新的坐标（扩展坐标的维度）将Z线性化：
$$
\begin{bmatrix}
x\\y
\end{bmatrix}
\Leftarrow\Rightarrow
\begin{bmatrix}
\hat{x}\\
\hat{y}\\
\hat{z}
\end{bmatrix}=
\begin{bmatrix}
f &0&0&0\\
0&f&0&0\\
0&0&1&0
\end{bmatrix}
\begin{bmatrix}
X\\
Y\\
Z\\
1
\end{bmatrix}
$$
坐标$(\hat{x},\hat{y},\hat{z})$就是像点$p=(x,y)$的齐次坐标,其中
$$
\begin{cases}
x=\frac{\hat{x}}{\hat{z}}\\
y=\frac{\hat{y}}{\hat{z}}\\
\hat{z}\neq0
\end{cases}
$$
可见通过扩展坐标维度构建其次坐标的步骤就是：

1. 将x,y同时除以一个不为0的$\hat{z}$
2. 并将$\hat{z}$作为其添加维度的坐标

通常选择$\hat{z}=1$

## 3.Perspektivische Projektion mit kalibrierter Kamera

Perspective Projection透视投影 with a Calibrated标定的 Camera

### 3.1内参数

- 相机的内参数由2部分组成：

  1. 射影变换本身的参数，相机的焦点到成像平面的距离，也就是焦距f

  2. 从成像平面坐标系到**像素坐标系**的转换。

     - 像素坐标系的原点在左上角
       - ->原点的平移

     - 像素是一个矩形块: 假设其在水平和竖直方向上的长度为$\alpha和\beta$
       - ->坐标的缩放

- 若像素坐标系的水平轴为u，竖轴为v；原点平移了$(c_x,c_y)$，那么成像平面点(x,y)在像素坐标系下的坐标为：
  $$
  u=\alpha\cdot x+c_x\\
  v=\beta\cdot y+c_y\tag{2}
  $$
  将1.Bildentstehung中的1式代入这里的2式可得：
  $$
  \begin{cases}
  u=\alpha\cdot f\frac{X}{Z}+c_x\\
  v=\beta\cdot f\frac{Y}{Z}+c_y
  \end{cases}\Rightarrow
  \begin{cases}
  u=f_x\frac{X}{Z}+c_x\\
  v=f_y\frac{Y}{Z}+c_y
  \end{cases}\tag{3}
  $$
  将3式其改写成齐次坐标：
  $$
  \begin{bmatrix}
  u\\
  v\\
  1
  \end{bmatrix}=\frac{1}{Z}
  \begin{bmatrix}
  \alpha & \theta & c_x\\
  0 & \beta & c_y \\
  0 & 0 &1
  \end{bmatrix}
  \begin{bmatrix}
  f & 0 &0\\
  0 & f & 0\\
  0 & 0 & 1
  \end{bmatrix}
  \begin{bmatrix}
  X\\
  Y\\
  Z
  \end{bmatrix}
  =\frac{1}{Z}
  \begin{bmatrix}
  f_x & f_\theta &c_x\\
  0 & f_y & c_y\\
  0 & 0 & 1
  \end{bmatrix}
  \begin{bmatrix}
  X\\
  Y\\
  Z
  \end{bmatrix}\tag{4}
  $$
  由此得到**内参数矩阵(Camera Intrinsics) K**:
  
  也叫Calibration Matrix标定矩阵
  $$
  K=K_sK_f
  =
  \begin{bmatrix}
  f_x & f_\theta &c_x\\
  0 & f_y & c_y\\
  0 & 0 & 1
  \end{bmatrix} \tag{5}
  $$
  
  - 这里的$\theta$是：像素形状不是矩形而是平行四边形时倾斜的角度。通常像素形状都是矩阵=0。
  
  转为非齐次：
  $$
  K\Pi_0=\begin{bmatrix}
  f_x & f_\theta &c_x\\
  0 & f_y & c_y\\
  0 & 0 & 1
  \end{bmatrix}
  \begin{bmatrix}
  1 & 0 & 0&0\\
  0&1&0&0\\
  0&0&1&0
  \end{bmatrix}=\begin{bmatrix}
  f_x & f_\theta &c_x &0\\
  0 & f_y & c_y & 0 \\
  0 & 0 & 1 &0
  \end{bmatrix}
  $$
  
  - $K_f$：Focal Length Matrix。世界坐标系->成像平面坐标系
  - $\Pi_0$：Generic Projection Matrix。将齐次转换为非齐次矩阵
  - $K_s$：Pixel Matrix。成像平面坐标系->像素平面坐标系
  
- 由5式可知K由4个相机构造相关的参数有关

  - $f_x,f_y$：和相机的焦距,像素的大小有关

    $f_x=\alpha\cdot f;f_y=\beta\cdot f$

  - $c_x,c_y$：是平移的距离，和相机成像平面的大小有关

  求解相机内参数的过程称为**标定**

### 3.2外参数

- 由3.1的4式可得：$p=KP$

  - p：是成像平面中像点在像素坐标下的坐标
  - K：是内参数矩阵
  - P：是相机坐标系下的空间点P的坐标。（P的像点是p）

  但相机坐标系是会随相机移动而动的，不够稳定，为了稳定需要引入**世界坐标系**。

  SLAM中的视觉里程计就是在求解相机在世界坐标系下的运动轨迹。

- 相机坐标系$P_c$转为世界坐标系$P_w$ :$P_c=RP_w+t$

  - R：是旋转矩阵
  - t：是平移矩阵

  $$
  P_c=
  \begin{bmatrix}
  R & t\\
  0^T &1
  \end{bmatrix}P_w
  $$

  由此可得**外参数(Camera Extrinsics)T **:
  $$
  T=\begin{bmatrix}
  R & t\\
  0^T &1
  \end{bmatrix}
  $$

### 3.3 相机矩阵

将内外参数组合到一起称之为**相机矩阵**，用于将真实场景中的三维点投影到二维的成像平面。
$$
\begin{bmatrix}
u\\
v\\
1
\end{bmatrix}
=\frac{1}{Z}K_sK_f\Pi_0P_c
=\frac{1}{Z}
\begin{bmatrix}
f_x & f_\theta &c_x &0\\
0 & f_y & c_y & 0 \\
0 & 0 & 1 &0
\end{bmatrix}
\begin{bmatrix}
R & t\\
0^T &1
\end{bmatrix}
\begin{bmatrix}
X_W\\
Y_W\\
Z_W\\
1
\end{bmatrix}\tag{4}
$$



## 4. Bild,Urbild und Cobild

Image,Preimage原相and Coimage余相

### 4.1 直线在空间的表示

齐次坐标系下: 以方向V经过点$P_0$的一条直线。
$$
L^{(hom)}=\{P_0^{(hom)}+\lambda[v_1,v_2,v_3,0]^T\ |\ \lambda\in R   \}
$$

### 4.2 图像,原相和余相

- The image of a point and of a line,respectively分别的，is their perspective projection透视投影:$\Pi_oP^{(hom)}$和$\Pi_0L^{(hom)}$

- Preimage原相：

  - The Preimage of a point P is all the points in space which project onto a single image point in the image plane
    - 所以点的原相是经过原点的一条直线
  - The Preimage of a line L is all the points which project onto a single line in the image plane
    - 所以线的原相是经过原点的一个平面

- Coimage余相：

  - The Coimage of points or lines is the orthogonal complement正交补 of the preimage.
    - 点的余相是一个平面上所有的向量，这个平面与点的原相（直线）垂直。
    - 线的余相是一个向量，与线的原相（平面）垂直
  

### 4.3 一些有用的性质

- 成像平面上的一条直线L一般使用余相Coimage来表示：
  - 余相向量：$l\in R^3$, 直线的点$x$ 显然也在直线的原相上，因为余相垂直于原相，所以：
  - $x^Tl=l^Tx=0$
- 共线性Collinearity:
  - 图像上的点$x_1,x_2...x_n$是共线的
    - 如果$Rang([x_1,..,x_n])\leq 2$
    - 如果对任意$w_i都>0$时， $M=\sum_{i=1}^n\omega_ix_ix_i^T$最小的特征值等于0
  - 三个图像上的点是共线的，如果$det[x_1,x_2,x_3]=0$
  - 但实际中由于Discretization离散化、Noise噪音。上面=0的条件是不可能达到的。需要利用thresholds阈值。



# *、 TUM CV课作业代码

## 第一次作业

### 1.1Color Image conversion

```matlab
function gray_image = rgb_to_gray(input_image)
    % This function is supposed to convert a RGB-image to a grayscale image.
    % If the image is already a grayscale image directly return it.
    
    if(size(input_image,3)~=3)
        gray_image = input_image;
        return
    end
    
    input_image = double(input_image)
    gray_image = 0.299*input_image(:,:,1) + 0.587*input_image(:,:,2) + 0.114*input_image(:,:,3);
    gray_image = uint8(gray_image);
end
```

### 1.2Image Gradient(索伯算子)

```matlab
function [Fx, Fy] = sobel_xy(input_image)
    % In this function you have to implement a Sobel filter 
    % that calculates the image gradient in x- and y- direction of a grayscale image.
    gx = [-1,0,1;-2,0,2;-1,0,1];
    gy = gx';
    
    Fx = filter2(gx,input_image,'same');
    Fy = filter2(gy,input_image,'same');
    
end
```

### 1.3 Harris Detector

- 基础方法：

    ```matlab
    function features = harris_detector(input_image, varargin)
        % In this function you are going to implement a Harris detector that extracts features
        % from the input_image.
    
    
    
        % *************** Input parser ***************
        % 创建具有默认属性值的输入解析器对象，用于检测输入进来的参数是否符合要求。
        %使用 `inputParser` 对象，用户可以通过创建输入解析器模式来管理函数的输入。要检查输入项，您可以为所需参数、可选参数和名称-值对组参数定义验证函数。还可以通过设置属性来调整解析行为，例如如何处理大小写、结构体数组输入以及不在输入解析器模式中的输入。
        P = inputParser;
        
        % addOptional:将可选的位置参数添加到输入解析器模式中
        P.addOptional('segment_length', 15, @isnumeric);
        P.addOptional('k', 0.05, @isnumeric);
        P.addOptional('tau', 1e6, @isnumeric);
        P.addOptional('do_plot', false, @islogical);
    	
    	% 解析函数输入
        P.parse(varargin{:});
    	
    	% results:指定为有效输入参数的名称和对应的值，以结构体的形式存储。有效输入参数是指名称与输入解析器模式中定义的参数匹配的输入参数。Results 结构体的每个字段对应于输入解析器模式中一个参数的名称。parse 函数会填充 Results 属性
        segment_length  = P.Results.segment_length;
        k               = P.Results.k;
        tau             = P.Results.tau;
        do_plot         = P.Results.do_plot;
        
        
        
        % *************** Preparation for feature extraction ***************
        % Check if it is a grayscale image，处理过的灰度照片是1维的，所以检测图片的维度
        if(size(input_image,3)==3)
            error("Image format has to be NxMx1");
        end
        
        % Approximation of the image gradient
        input_image = double(input_image);	% matlab读入图像的数据是uint8，而matlab中数值一般采用double型（64位）存储和运算
        [Ix,Iy] = sobel_xy(input_image);	% 由1.2的函数得到横向及纵向边缘检测的图像灰度值
        
        % Weighting
        % fspecial()创建一个高斯低通滤波来提高我们选取的窗口segment中间像素的影响力
        % 第一个参数指明用的是高斯滤波器。高斯滤波器是一种线性滤波器，能够有效的抑制噪声，平滑图像。
        % 第二个参数是这个滤波器的尺寸，和我们的窗口尺寸保持一致。
        % 第三个参数是sigma标准差。调整σ实际是在调整周围像素对当前像素的影响程度，调大σ即提高了远处像素对中心像素的影响程度，滤波结果也就越平滑。这里设置为segment_length/5
        w = fspecial('gaussian',[segment_length,1],segment_length/5);
        
        % Harris Matrix G
        % conv2(u,v,A,shape)先算矩阵A每列和u的卷积，再算矩阵A每列和v的卷积。same表示返回卷积中大小与A相同的部分，还有full（默认，全部）和valid（没有补零边缘的卷积部分）2个shape选项。
        % 这里是在计算图像中每一个点的哈里斯矩阵。
        G11 = double(conv2(w,w,Ix.^2,'same'));
        G22 = double(conv2(w,w,Iy.^2,'same'));
        G12 = double(conv2(w,w,Ix.*Iy,'same'));
        
        
        
        % *************** Feature extraction with the Harris measurement ***************
        % 首先根据公式H=det(G) - k*tr(G)*tr(G)计算 H
        H = ((G11.*G22 - G12.^2) - k*(G11 + G22).^2);
        % ceil(X)将 X 的每个元素四舍五入到大于或等于该元素的最接近整数
        % H是一个长方形矩阵，肯定会把边缘点包含进来。我们要消除边缘点，于是从长方形四边往里取一个segment_length/2边界。这个边界里的就是我们要取特征的部分，令mask为1。边界外的就令mask=0，乘上H后就自动消除边缘了。
        mask=zeros(size(H));
        mask((ceil(segment_length/2)+1):(size(H,1)-ceil(segment_length/2)),(ceil(segment_length/2)+1):(size(H,2)-ceil(segment_length/2)))=1;
        R = H.*mask;
    	% harris detector是用于找角点的，只要大于一定阈值的才是角点，所以小于阈值的令为0
        R(R<tau)=0;
        % find会返回R中非0的坐标
        [row,col] = find(R); 
        corners = R;
        % 注意matlab中图像用矩阵表示所以原点在左上方，而我们平时说的坐标原点在左下方，所以这里的col和row是颠倒的。
        features = [col,row]'; 
        
        
        % *************** Plot ***************
        % 将找到的特征滑倒图片里去
        if P.Results.do_plot == true
            figure
            imshow(input_image);
            hold on
            plot(features(1,:),features(2,:),'*');
        end
    end
    ```

- 进阶方法：

  前面普通方法用了一个全局的阈值threshold：tau。但以一张图片的特征分布变化是很大的，用一个全局的阈值就会导致特征过多，所以先用一个小阈值来消除一些弱特征来消除噪声是很有必要的。

  - 一个辅助函数

    产生一个矩阵，离这个矩阵中心点的距离如果小于min_dist那就是0，如果大于就是1.

    ```matlab
    function Cake = cake(min_dist)
        % The cake function creates a "cake matrix" that contains a circular set-up of zeros
        % and fills the rest of the matrix with ones. 
        % This function can be used to eliminate all potential features around a stronger feature
        % that don't meet the minimal distance to this respective feature.
        %[X,Y] = meshgrid(x,y) 基于向量 x 和 y 中包含的坐标返回二维网格坐标。X 是一个矩阵，每一行是 x 的一个副本；Y 也是一个矩阵，每一列是 y 的一个副本。
        [X,Y]=meshgrid(-min_dist:min_dist,-min_dist:min_dist);
        % 这样sqrt(X.^2+Y.^2)就相当于每个点到矩阵中心的距离
        Cake=sqrt(X.^2+Y.^2)>min_dist;
        
    end
    ```

  - 

  ```matlab
  function [min_dist, tile_size, N] = harris_detector(input_image, varargin)
      % In this function you are going to implement a Harris detector that extracts features
      % from the input_image.
      
      % *************** Input parser ***************
      % 相比普通方法增加如下变量
      % 1. min_dist:两个特征之间的最小距离，单位是pixel
      % 2. tile_size:定义了一个局部区域的大小
      % 3. N:一个局部区域里最多可以有的特征数量
      P = inputParser;
      P.addOptional('segment_length', 15, @isnumeric);
      P.addOptional('k', 0.05, @isnumeric);
      P.addOptional('tau', 1e6, @isnumeric);
      P.addOptional('do_plot', false, @islogical);
      P.addOptional('min_dist', 20, @isnumeric);
      P.addOptional('tile_size', [200,200], @isnumeric);
      P.addOptional('N', 5, @isnumeric);
      P.parse(varargin{:});
      segment_length  = P.Results.segment_length;
      k               = P.Results.k;
      tau             = P.Results.tau;
      tile_size       = P.Results.tile_size;
      N               = P.Results.N;
      min_dist        = P.Results.min_dist;
      do_plot         = P.Results.do_plot;
      
      if size(tile_size) == 1
          tile_size   = [tile_size,tile_size];
      end
      % *************** Preparation for feature extraction ***************
      % 与普通方法一致
      if(size(input_image,3)==3)
          error("Image format has to be NxMx1");
      end
      input_image = double(input_image);	
      [Ix,Iy] = sobel_xy(input_image);	
      w = fspecial('gaussian',[segment_length,1],segment_length/5);
      G11 = double(conv2(w,w,Ix.^2,'same'));
      G22 = double(conv2(w,w,Iy.^2,'same'));
      G12 = double(conv2(w,w,Ix.*Iy,'same'));
      
      
      
      % *************** Feature extraction with the Harris measurement ***************
      % 与普通方法一致
      H = ((G11.*G22 - G12.^2) - k*(G11 + G22).^2);
      mask=zeros(size(H));
      mask((ceil(segment_length/2)+1):(size(H,1)-ceil(segment_length/2)),(ceil(segment_length/2)+1):(size(H,2)-ceil(segment_length/2)))=1;
      R = H.*mask;
      R(R<tau)=0;
      [row,col] = find(R); 
      corners = R;
      
      % *************** Feature preparation ***************
      % 扩展我们的角点矩阵：在corners角点矩阵周围扩展一圈宽度为min_dist的0
      % 具体做法是，先建那么大的矩阵，然后中间corners部分赋值进去。
      expand = zeros(size(corners)+2*min_dist);
      expand(min_dist+1:min_dist+size(corners,1),min_dist+1:min_dist+size(corners,2)) = corners;
      corners = expand;  
      % [B,I] = sort(A,direction)按照direction(ascend/descend)来对矩阵A的每一列进行排序.
      % B是排列完的矩阵(mxn,1)维的，I是排序完的B在A中的索引，即B==A(I)
      [sorted, sorted_index] = sort(corners(:),'descend');
      % 排除角点矩阵范围内那些为0的点，因为他们不是特征点
      sorted_index(sorted==0)=[];
      
      % *************** Accumulator array ***************
  	% 创建一个矩阵acc_array用于储存每个小部分teil里的参数。所以共有size(image)/tile_size个小tile.
      acc_array = zeros(ceil(size(input_image,1)/tile_size(1)),ceil(size(input_image,2)/tile_size(2)));
      % 上面排序后我们得到sorted_index个特征。此外每个teil最多有N个特征。这两个共同影响了我们最后特征应该有多少个。
      features = zeros(2,min(numel(acc_array)*N,numel(sorted_index)));
      
      % *************** Feature detection with minimal distance and maximal number of features per tile ***************
      % 首先创建一个矩阵，大于矩阵中心距离min_dist的元素设置为1，即中间的各个元素都是0
      % 这样一个特征点周围的矩阵 点乘上 这个cake矩阵后，就保证了这个特征点最小距离内没有其他特征点。
      Cake = cake(min_dist);
      count = 1;
      for i = 1:numel(sorted_index)
      	% 前面排列时根据特征强度降序排列，所以我们先处理特征最强的。因为由于我们设置的最大特征数，导致一些特征丢失，那肯定丢失那些特征弱的。
      	% sort返回的是(mxn,1)维矩阵，index也指的是第几个，所以下面还要手动转化为x,y
          current = sorted_index(i);
          % 如果=0说明不是特征点（角点）
          if corners(current) == 0
              continue;
          else 
          	% Y = floor(X) 将 X 的每个元素四舍五入到小于或等于该元素的最接近整数。相反于ceil()		
          	% 因为index是1维的，所以要手动转化为x,y,注意这里原点在左上角，所以x表示列数
              % 对索引除以行数后，就得到除尽了几列，那再加一列就是元素矩阵中的列数x.
              x_corners = floor(current/size(corners,1));
              % 索引-除尽的列数*行数=元素在下一列的第几行
              y_corners = current-x_corners*size(corners,1);
              x_corners = x_corners+1;
          end
          	% 求得我们当前的特征点，是图像中的第几个小teil区域。同样注意这里原点在左上角，所以x表示列数
          	% 原本corners是覆盖了整个范围，然后我们扩展了一圈min_dist以方便清空特征点周围的特征。
          	% 所以现在corner的范围是原先corner+min_dist那一圈。
          	% min_dist那一圈都是0,我们在画小teil的时候不需要考虑这些min_dist。
          	% 所以要减掉min_dist坐标，以得到特征在原先corner下的坐标。
              x_acc = ceil((x_corners-min_dist)/tile_size(2));
              y_acc = ceil((y_corners-min_dist)/tile_size(1));
  			
  			% 这样一个特征点周围的矩阵 点乘上 这个cake矩阵后，就保证了这个特征点最小距离内没有其他特征点。
              corners(y_corners-min_dist:y_corners+min_dist,x_corners-min_dist:x_corners+min_dist) = corners(y_corners-min_dist:y_corners+min_dist,x_corners-min_dist:x_corners+min_dist).*Cake;
             
          % 如果小teil中的特征数没有超过一个teil的最大特征数N,那么就将特征加入进去   
          if acc_array(y_acc,x_acc) < N
              acc_array(y_acc,x_acc) = acc_array(y_acc,x_acc)+1;
              % 我们要得到原始图像下的坐标，而现在corner矩阵为了方便计算扩充了一圈，所以现在记录特征点要减去这一圈对应的坐标。
              features(:,count) = [x_corners-min_dist;y_corners-min_dist];
              count = count+1;
          end
      end
      % all(_,1)检测每一列并返回一个行矩阵，这里如果一列都是0元素，那行矩阵的该元素为1.
      % 消除所有的0列。
      features(:,all(features==0,1)) = []; 
      
      % *************** Plot ***************
      if P.Results.do_plot == true
          figure
          imshow(input_image);
          hold on
          plot(features(1,:),features(2,:),'*');
      end
  end
  ```
  

## 第二次作业

### 2.1 Point_Correspondence

计算2个图片之间的对应点：

```matlab
% 输入：I1,I2是2个灰度的图片
% Ftp1,Ftp2分别是2张图片所有的特征的坐标（x,y）维度是[2,N],N是特征个数
% varargin是可选的参数
function [window_length, min_corr, do_plot, Im1, Im2] = point_correspondence(I1, I2, Ftp1, Ftp2, varargin)
    % In this function you are going to compare the extracted features of a stereo recording
    % with NCC to determine corresponding image points.
    
    % *************** Input parser ***************
    % 检验可选参数的输入是否符合要求
    p = inputParser;
    addOptional(p,'window_length',25,@(x) isnumeric(x) && (mod(x,2)==1) &&(x>1));
    addOptional(p,'min_corr',0.95,@(x) isnumeric(x) &&(x>0) && (x<1));
    addOptional(p,'do_plot',false, @islogical);
    
    p.parse(varargin{:});
    
    window_length = p.Results.window_length;
    min_corr = p.Results.min_corr;
    do_plot = p.Results.do_plot;
    
    Im1 = double(I1)
    Im2 = double(I2)
    
    % *************** Feature preparation准备 ***************
    % 消除离边界太近的参数，并返回剩下参数的个数
    % 找到一个范围，在这个范围里的都要，在这个范围外的都舍弃
    range_x_begin  = ceil(window_length/2);
    range_x_end    = size(I1,2) - floor(window_length/2);   
    range_y_begin  = ceil(window_length/2);
    range_y_end    = size(I1,1) - floor(window_length/2);
    
    % 将在range外的特征值都变成1
   	% logical(A)将数据A中所有非0值都变成逻辑数1，这里ind1就会变成一个2XN的1/0逻辑数组
   	% 注意logical返回的是逻辑值！！！
	ind1 = logical([Ftp1(1,:)<range_x_begin; Ftp1(1,:)>range_x_end; Ftp1(2,:)<range_y_begin; Ftp1(2,:)>range_y_end]);
	% 将这个2XN的1/0逻辑数组变成1XN的1/0逻辑数组
	% any(ind1,1)后面的1表示检测第一个维度（列）如果有一个非0数就返回1。
    ind1 = any(ind1,1);
    % 1XN逻辑数组中1即代表这个特征值在数组外，令该特征值为空，即在数组中删除了这个值。
    % 加入C是一个逻辑数组[1,1,0,1],那么A(:,C)只会返回C中逻辑值为1的部分
    Ftp1(:,ind1) = [];   
    
    % 原理同上处理第二个图片
    ind2 = logical([Ftp2(1,:)<range_x_begin; Ftp2(1,:)>range_x_end; Ftp2(2,:)<range_y_begin; Ftp2(2,:)>range_y_end]);
    ind2 = any(ind2,1);
    Ftp2(:,ind2) = [];   
    
    no_pts1  = size(Ftp1,2);
    no_pts2  = size(Ftp2,2);
    
    % *************** Normalization ***************
    % 用一、4.的SSD方法将每一个窗口里的图片信息正则化，并将正则后的图片灰度强度按列存在Mat_feat_1/2中
   	% 一个窗口的大小，窗口的正中间是特征的坐标。每一个特征都有一个自己的窗口
    win_size = -floor(window_length/2):floor(window_length/2);
    % 每一个特征的窗口里的正则值都按一列保存在矩阵Mat_feat_1中
    Mat_feat_1 = zeros(window_length*window_length,no_pts1);
    Mat_feat_2 = zeros(window_length*window_length,no_pts2);
   
   	% 同时遍历2张图片所有的特征，因为2张图片的特征数不一致，所以取大的那个
    for i = 1:max(no_pts1,no_pts2)
        if i <= no_pts1
        	% win_size是一个类似[-2,-1,0,1,2]的1维矩阵，矩阵+1个值=矩阵每个元素加上这个值
        	% 加上特征的x,y坐标后，如x=1，y=1就会变成[-1,0,1,2,3]的一个窗口
            win_x_size = win_size + Ftp1(1,i);
            win_y_size = win_size + Ftp1(2,i);
            % 得到窗口的大小后，获得这个窗口内的图片的灰度强度值
            window = Im1(win_y_size,win_x_size);
            % 将这个窗口内的图片参数正则化
            Mat_feat_1(:,i) = (window(:)-mean(window(:)))/std(window(:));
        end
        
        if i <= no_pts2
            win_x_size = win_size + Ftp2(1,i);
            win_y_size = win_size + Ftp2(2,i);
            window = Im2(win_y_size,win_x_size);    
            Mat_feat_2(:,i) = (window(:)-mean(window(:)))/std(window(:));
    end
    
    % *************** NCC calculations ***************
    % 根据各个窗口中灰度强度，用一、4.的NCC方法的公式计算NCC矩阵，要注意2张图片的顺序
    NCC_matrix = 1/(window_length*window_length-1)*Mat_feat_2'*Mat_feat_1;
    % NCC矩阵中的每个值都是相关度
    % NCC矩阵中所有相关度小于最小阈值的值都设置为0
    NCC_matrix(NCC_matrix<min_corr) = 0;
    % 将相关度按降序排列
    % [B,I]=sort(A),B是A排列后的矩阵，I描述了 A 的元素沿已排序的维度在 B 中的排列情况，即索引
    % B=A(I)
    [sorted_list,sorted_index] = sort(NCC_matrix(:),'descend');
    % 那些小于最小阈值的值（即=0）丢弃
    sorted_index(sorted_list==0) = [];
    
    % *************** Correspondeces ***************
    % 用上面储存了2个图片各特征点之间的相关度的矩阵NCC_matrix和它的降值排列索引sorted_index来确定相关图像点corresponding image points
    
    % 将2张图片中2个相关的点保存在矩阵cor=[x1,y1,x2,y2]中，所以这里是4行。
    % 相关点是根据2个特征点的窗口内灰度强度的相关性确定的，所以是min(no_pts1,no_pts2)列
    cor = zeros(4,min(no_pts1,no_pts2));
    % 记录相关点的个数
    cor_num = 1;
    
    % numel(A)返回数组A中的元素数目
    for i = 1:numel(sorted_index)  
    	% NCC矩阵中所有相关度小于最小阈值的值都设置为0了，所以遇到这些特征点直接跳过就行
        if(NCC_matrix(sorted_index(i))==0)
            continue;
        else
        	% 找到我们现在要处理的2个特征点的相关度在NCC_matrix中的
        	% [row,col]=ind2sub(sz,ind)将线性索引转换为下标，返回数组row和col分别是大小为sz的矩阵的线性索引ind对应的等效行和列下标。
        	% sz 是包含两个元素的向量，其中 sz(1) 指定行数，sz(2) 指定列数
        	% 比如sz=[3 3],即一个3X3的矩阵。第二个线性索引ind=[2],那么row=2,col=1。可见线性索引顺序是竖着下来的。而矩阵下标是横着数过来的
            [Idx_fpt2,Idx_fpt1] = ind2sub(size(NCC_matrix),sorted_index(i));
        end
		% 如果一个相关点被找到，就将该列设为0，保证图片1中的1个特征只会映射图片2中的1个特征。
		% NCC_matrix中的x,y包含了第二个图片中的点X和第一个图片中的点Y之间的correlation相关性
        NCC_matrix(:,Idx_fpt1) = 0;
        % Ftp1,Ftp2分别是2张图片所有的特征的坐标（x,y）维度是[2,N],N是特征个数
        % 将图片1中第Idx_fpt1个特征点和图片2中第Idx_fpt2个特征点，加入我们的相关点
        cor(:,cor_num) = [Ftp1(:,Idx_fpt1);Ftp2(:,Idx_fpt2)];
        cor_num = cor_num+1;
    end
    % 因为cor初始化为min(no_pts1,no_pts2)列，所以后面有一堆为[0,0,0,0]的点，删掉他们。
    cor = cor(:,1:cor_num-1)
    
    % *************** Visualize the correspoinding image point pairs ***************
    if do_plot
    	% 创建图床窗口，窗口名字是Punkt-Korrespondenzen
        figure('name', 'Punkt-Korrespondenzen');
        % 用无符号整数显示灰度图片I1
        imshow(uint8(I1))
        hold on
        % 给第一张图中的相关点画上红星
        plot(cor(1,:),cor(2,:),'r*')
        imshow(uint8(I2))
        % 将上面的图片透明度设置为0.5
        alpha(0.5);
        hold on
        % 给第二张图中的相关点画上绿星
        plot(cor(3,:),cor(4,:),'g*')
        % 给两个相关点之间画上连线
        for i=1:size(cor,2)
            hold on
            x_1 = [cor(1,i), cor(3,i)];
            x_2 = [cor(2,i), cor(4,i)];
            line(x_1,x_2);
        end
        hold off
    end
end
```





