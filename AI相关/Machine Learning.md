# 一. Supervised Learning监督学习

- 两类问题

  1. Regression problems回归问题Given a picture of a person, we have to predict their age on the basis of the given picture

  2. Classification problems分类问题

     Given a patient with a tumor, we have to predict whether the tumor is malignant or benign

## 1. Regression

### (1)一元线性规划

- Hypothesis假设h

  ![](https://github.com/Fernweh-yang/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/supervised%20learning%20model.png?raw=true)

  :比如对于linear Regression线性规划: 

$$
h_\theta(x)=\theta_0+\theta_1x\tag{1}
$$

- cost function:

  解决如何确定$\theta$的问题。
  
  We can measure the accuracy of our hypothesis function by using a **cost function**
  $$
  J(\theta_0,\theta_1)=\frac{1}{2m}\sum_{i=1}^m(\hat{y}_i-y_i)^2=\frac{1}{2m}\sum_{i=1}^m(h_\theta(x_i)-y_i)^2\tag{2}
  $$
  目标是最小化这个cost function。
  
- Gradient descent梯度下降

  不仅可用于最小化cost function J，也可用于最小化其他函数。

  - 通用算法：

    repeat until convergence(不断迭代至收敛): 

  $$
  \theta_j:=\theta_j-\alpha\frac{\partial}{\partial\theta_j}J(\theta_0,\theta_1), \quad for\: j=0\: and\:  j=1\tag{3}\\
  $$
  
   		这里$\alpha$是learning rate
  
  - Subtlety微妙之处：Simultaneous同时发生的。这里两个参数的新值都是基于旧值计算的，两个都算完了再同时代入下一个迭代。
  - learning rate如果
    - 太小，就会导致收敛速度太慢
    - 太大，就可能会导致错过收敛点

### (2)Multivariate Linear Regression多元线性规划

- Hypothesis假设h
  $$
  h_\theta(x)=\theta_0+\theta_1x_1+\theta_2x_2+...+\theta_nx_n=[\theta_0\;\theta_1\;...\;\theta_n]\begin{bmatrix}x_0\\x_1\\\vdots\\x_n\end{bmatrix}
  $$

​		说明：$x_j^{(i)}$=第i个训练样本的第j个参数。

​					$x^{(i)}$=第i个训练样本的输入参数们。

- Cost function:
  $$
  J(\theta)=\frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2
  $$

- Gradient descent:

  repeat until convergence{
  $$
  \begin{aligned}
  \theta_j:&=\theta_j-\alpha\frac{\partial}{\partial\theta_j}J(\theta)\\
  &= \theta_j-\alpha\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}\qquad for\ j:=0\ \cdots n
  \end{aligned}
  $$
  }

  - Feature Scaling特征缩放:Mean Normalization均值归一化

    因为不同特征feature的变化的范围不同，所以使特征统一标准化。这样也可以加速梯度函数收敛。
    $$
    x_i=\frac{x_i-\mu_i}{s_i}
    $$
    其中：$\mu_i$：是所有特征值$x_i$的平均值。

    ​            $s_i$：是特征值变化的范围（最大值-最小值）

- Polynomial Regression多项式回归

  如果我们的Hypothesis函数以一条直线不能很贴合数据时，可以通过加二次quadratic，三次cubic，平方根square root项等来改变曲线的形状，比如：
  $$
  h_\theta(x)=\theta_0+\theta_1x_1+\theta_2\sqrt{x_1}+\theta_3x_1^3
  $$
  

### (3)Normal Equation

另一种让cost function最小化的方法。

- 与Gradient descent对比

  - 上面用到的Gradient descent是通过选择一个学习速率并用迭代的形式来求各个参数$\theta$。

  - Normal Equation是用数学analytically分析地来求解各个参数。
  - Gradient descent适合有很多特征x(i)的时候，Normal Equation在处理很多特征时速度很慢。

- Formula:

  要让cost function J最小，即让$J(\theta)$的导数=0，也即$y=X\theta$

  因为只有方阵才有逆，但我们的特征$x^{(i)}$的个数n和训练样本数m基本不可能相等，所以X不是方阵。为了求解上面的等式，要先让X乘它的转置$X^T$来使他们变成方阵，然后再乘逆。最终得到解：
  $$
  \theta=(X^TX)^{-1}X^Ty
  $$

- 如果$X^TX$noninvertible不可逆

  原因有2：

  1. Redundant Features:即不同特征之间是线性依赖的。

     比如特征x1是米的平方，特征x2是英尺的平法。他们之间只差一个倍数，是linearly dependent的。

  2. Too many features: 即特征x(i)的数目n多过了训练集的样本数m。

  解决方法是

  1. 删除一些features
  2. 或使用regularization

### (4)Overfitting Problem

- 如果我们的hypothesis方程不正确会导致两类问题：

  - **Underfitting**: the form of our hypothesis function h maps poorly to the trend of the data.
    - It is usually caused by a function that is too simple or uses too few features

  - **Overfitting**: a hypothesis function that fits the available data but does not generalize well to predict new data

    -  It is usually caused by a complicated function that creates a lot of unnecessary curves and angles unrelated to the data.

  - 解决方法：

    Reduce the number of features:

    - Manually select which features to keep.
    - Use a model selection algorithm

    Regularization正则化：

    - Keep all the features, but reduce the magnitude of parameters $\theta_j$
    - Regularization works well when we have a lot of slightly useful features.

- Cost function

  如果hypothesis function的参数过多，我们需要消去多余的参数$\theta$。不需要真的删除，只需要修改cost function让多余的参数$\theta$得到inflate膨胀。这样我们在用gradient descent 来求$\theta$使cost function最小化时，会自动的让那些参数趋近于0

  - 比如：我们的 h(z)的z是：$\theta_0+\theta_1x+\theta_2x^2+\theta_3x^3+\theta_4x^4$，然后想要让他的3次，4次参数失效。

    ​			只需修改cost function为: $min_\theta\frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2+1000\theta_3^2+1000\theta_4^2$

  - 一个总结性的公式：

  $$
  min_\theta\frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2+\lambda\sum_{j=1}^{n}\theta_j^2
  $$

  - 上式的$\lambda$叫做regularization parameter。
    - 如果太小没效果
    - 如果太大，会使得hypothesis function太过smooth,从而导致underfitting。

- Regularized Linear Regression

  - Gradient Descent:

    我们不希望惩罚penalize $\theta_0$所以单独列出来。

    下面的$\frac{\lambda}{m}\theta_j$代表了我们的正则化

    $$
    \begin{aligned}
      Repeat\{\\
      &\theta_0:= \theta_0-\alpha\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})x_0^{(i)}\\
      &\theta_j:= \theta_j-\alpha[(\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)})+\frac{\lambda}{m}\theta_j]\quad j\in\{1,2..n\}\\ 
      \}
      \end{aligned}
    $$

  - Normal Equation:
    
    $\lambda L$项代表了我们的正则化
    $$
    \theta=(X^TX+\lambda L)^{-1}X^Ty\\
    其中L= \begin{bmatrix}
       0 &  &  &\\
        & 1 &  &\\
        &  & \ddots &\\
        &  & &1
      \end{bmatrix}
    $$
    

## 2. Classification

- To attempt classification, one method is to use linear regression and map all predictions greater than 0.5 as a 1 and all less than 0.5 as a 0. However, this method doesn't work well because classification is not actually a linear function.

### (1)Logistic Regression逻辑回归

- Hypothesis Representation

	如上所说我们可以用线性规划的方法去尝试做分类问题，但这并不是一个好方法。因为当我们知道$y\in \{0,1\}$时，$h_\theta(x)$的值大于1或者小于0时是没有意义的。所以我们可以用另一种形式的假设：
	
	- Logistic Function ,也叫做Sigmoid Function
	  $$
	  \begin{cases}
	  g(z)=\frac{1}{1+e^{-z}}\\
	  z=\theta^Tx
	  \end{cases}\quad\Rightarrow\quad
	  h_\theta(x)=g(\theta^Tx) =\frac{1}{1+e^{-\theta^Tx}}
	  $$
	
	- $h_\theta(x)$给了输出y=1的概率
	  $$
	  h_\theta(x)=P(y=1|x;\theta)=1-P(y=0|x;\theta)
	  $$
	  比如$h_\theta(x)$=0.7就是当前输入x有70%的概率得到输出y=1。
	  
	- 求出某个类别的概率h(x)后，如果它大于等于0.5则分类为1，如果小于0.5则分类为0。
	
- Decision Boundary决策边界

  - The **decision boundary** is the line that separates the area where y = 0 and where y = 1. It is created by our hypothesis function的$\theta^Tx$

  - 由上可知h大于等于0.5则分类为1。此时z应该>0。

  - 由上可知h小于0.5则分类为0。此时z应该<0。
  
  - 决策边界可以不是线性的
  
    即我们的sigmoid function g(z)不必是线性的。比如它可以是个圆$z=\theta_0+\theta_1x_1^2+\theta_2x_2^2$
  
- Cost function

  我们不能使用和Linear regression一样的cost function，因为这会导致我们的cost function是non convex的，即函数的图像不是先单降后单升的凸型而是有很多局部最小的波浪形wavy。

  Logistic Regression为了得到convex function需要用下面这样的cost function:
  $$
  \begin{aligned}
  &J(\theta)=\frac{1}{m}\sum_{i=1}^mCost(h_\theta(x^{(i)}),y^{(i)})\\
  &Cost(h_\theta(x),y)=-log(h_\theta(x))\quad \quad &if\ y=1\\
  &Cost(h_\theta(x),y)=-log(1-h_\theta(x))\quad \quad &if\ y=0
  \end{aligned}
  $$
  自己画一下$h_\theta \in[0,1]$的Cost图像可发现：

  1. 如果正确解y=1，那么当$h_\theta$也为1时cost function将为0。当$h_\theta$接近0时cost function将趋于无穷大。
  2. 如果正确解y=0，那么当$h_\theta$也为0时cost function将为0。当$h_\theta$接近1时cost function将趋于无穷大。

  由此使得我们的cost function是convex function了(一个u型的函数)

  - 上面的方程可以合并为
    $$
    \begin{aligned}
    &Cost(h_\theta(x),y)=-ylog(h_\theta(x))-(1-y)log(1-h_\theta(x))\\
    &J(\theta)=\frac{1}{m}\sum_{i=1}^m[y^{(i)}log(h_\theta(x^{(i)}))+(1-y^{(i)})log(1-h_\theta(x^{i}))]
    \end{aligned}
    $$
    继续将其向量化：
    $$
    \begin{aligned}
    &h=g(X\theta)\\
    &J(\theta)=\frac{1}{m}(-y^Tlog(h)-(1-y)^Tlog(1-h))
    \end{aligned}
    $$

- Gradient Descent

  找到一组$\theta$使得cost function最小化
  $$
  \begin{aligned}
  Repeat&\{\\
  &\theta_j:=\theta_j-\alpha\frac{\partial}{\partial\theta_j}J(\theta)\\
  &\theta_j:=\theta_j-\frac{\alpha}{m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}\\
  \}
  \end{aligned}
  $$
  向量化：
  $$
  \theta:=\theta-\frac{\alpha}{m}X^T(g(X\theta)-\vec{y})
  $$
  
- Advanced Optimization

  除了Gradient Descent还有"Conjugate gradient", "BFGS", and "L-BFGS" 也可以用来找$\theta$。但他们更加的sophisticated复杂也更加的快速。

  有很多现成的库可以实现这些算法。

### (2)Multiclass classification

- One vs all

  - 上面的logistic regression的分类只有2个y={1,2}。现在有更多的种类y={0,1,...n}。

  - 所以我们就依次的把某一类和其他类作为两个类 y={1,rest}, 来不停的求1的logistic regression.最后predict新的数据集x时，选择能最大化Hypothesis那一组类。预测的种类y即它。
    $$
    y\in\{0,1...,n\}\\
    h_\theta^{(0)}(x)=P(y=o|x;\theta)\\
    h_\theta^{(1)}(x)=P(y=1|x;\theta)\\
    ...\\
    h_\theta^{(n)}(x)=P(y=n|x;\theta)\\
    prediction=max(h_\theta^{(i)}(x))
    $$
    

### (3)Overfitting Problem

- Cost function

  在我们的Regularized Logistic原有cost function后加上$\frac{\lambda}{2m}\sum_{j=1}^n\theta_j^2$。代表忽视某一个参数
  $$
  J(\theta)=\frac{1}{m}\sum_{i=1}^m[y^{(i)}log(h_\theta(x^{(i)}))+(1-y^{(i)})log(1-h_\theta(x^{i}))]+\frac{\lambda}{2m}\sum_{j=1}^n\theta_j^2
  $$

- Gradient descent
  $$
  \begin{aligned}
  Repeat&\{\\
  &\theta_0:=\theta_0-\frac{\alpha}{m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_0^{(i)}\\
  &\theta_j:=\theta_j-\alpha[\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}+\frac{\lambda}{m}\theta_j]\\
  \}
  \end{aligned}
  $$
  

# 二. Unsupervised Learning无监督学习

- 两个例子

  1. Clustering:

     Take a collection of 1,000,000 different genes, and find a way to automatically group these genes into groups that are somehow similar or related by different variables, such as lifespan, location, roles, and so on.

  2. Non-Clustering:

     The "Cocktail Party Algorithm", allows you to find structure in a chaotic environment. (i.e. identifying individual voices and music
