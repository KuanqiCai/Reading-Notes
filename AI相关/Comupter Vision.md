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
      % B是排列完的矩阵，I是排序完的B在A中的索引，即B==A(I)
      [sorted, sorted_index] = sort(corners(:),'descend');
      % 排除角点矩阵范围内那些为0的点，因为他们不是特征点
      sorted_index(sorted==0)=[];
      
  end
  ```

  

