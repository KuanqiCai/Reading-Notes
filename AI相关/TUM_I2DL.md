# *. 各种库的安装使用

## 0) 用GPU跑jupyter notebook

https://thegeeksdiary.com/2021/10/07/how-to-setup-tensorflow-with-gpu-support-in-windows-11/

## 1）python os库

https://www.runoob.com/python/os-file-methods.html

## 2） python numpy库

https://numpy.org/doc/stable/

### 2.1array

- 固定的一维矩阵`a=np.array([2,3,4])`
- 固定的二维矩阵`a=np.array([2,3,4],[1,2,3])`
- 随机的mxn维矩阵`a=np.random.rand(4,3)`



# *. 一些特别的Python机制

## 1） Iterator迭代器和Generator生成器

- 为什么需要迭代器和生成器？

  - 受到内存限制，列表容量肯定是有限的。而且，创建一个包含100万个元素的列表，不仅占用很大的存储空间，如果我们仅仅需要访问前面几个元素，那后面绝大多数元素占用的空间都白白浪费了。
  - 如果列表元素可以按照某种算法推算出来，就可以在循环的过程中不断推算出后续的元素。这样就不必创建完整的list，从而节省大量的空间。
  - 迭代器和生成器就是Python中这种一遍循环一边计算的机制
  - Generator是一种Iterator，

- 类中实现2种方法：`__iter__()`和`__next__()`的就是iterator迭代器

  ```python
  class MyNumbers:
    # __iter__() 方法返回一个特殊的迭代器对象， 这个迭代器对象实现了 __next__() 方法并通过 StopIteration 异常标识迭代的完成。
    def __iter__(self):
      self.a = 1
      return self
    # __next__() 方法会返回下一个迭代器对象
    def __next__(self):
      if self.a <= 20:
        x = self.a
        self.a += 1
        return x
      else:
        raise StopIteration	# StopIteration 异常用于标识迭代的完成，防止出现无限循环的情况
  ```

- 生成器本质上还是一个迭代器，但它的实现更加的简洁

​	有2种创建Generator的方式:

  1. 用表达式生成：

     用**iter()**或把一个列表生成式的**[]改成()**，就创建了一个generator：

     ```python
     >>> L = [x * x for x in range(10)]			# List 2种写法
     >>> L = list(x * x for x in range(10))
     >>> L
     [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
     >>> g = iter(L)								# 用iter将list转为生成器
     >>> g = (x * x for x in range(10))			# g 是一个生成器
     >>> g
     <generator object <genexpr> at 0x104feab40>
     
     # generator 不能直接打印要用g.next()或循环来输出
     for i in g:
         print(i)
     ```

  2. 用带yield语句函数生成：

     如果一个函数定义中包含yield关键字，那么这个函数就不再是一个普通函数，而是一个generator：

      ```python
      # 普通函数：
      def fib(max):
          n, a, b = 0, 0, 1
          while n < max:
              print (b)					# print语句
              a, b = b, a + b
              n = n + 1
     
      # 生成器：
      def fib(max):
          n, a, b = 0, 0, 1
          while n < max:
              yield b						# yield语句
              a, b = b, a + b
              n = n + 1
      ```
     
     - 在调用生成器运行的过程中，每次遇到 yield 时函数会暂停并保存当前所有的运行信息，返回 yield 的值, 并在下一次执行 next() 方法时从当前位置继续运行。



# *.一些重要概念

## 0）超参数Hyperparameters

- 参数和超参数：

  - 参数：就是模型可以根据数据可以自动学习出的变量就是参数。
    - 比如，深度学习的权重，偏差等

  - 超参数：用来确定模型的一些参数
    * Network architecture
        * Choice of activation function
        * Number of layers
        * ...
    * Learning rate
    * Number of epochs
        * epoch数是一个超参数，它定义了学习算法在整个训练数据集中的工作次数。一个Epoch意味着训练数据集中的每个样本都有机会更新内部模型参数。
    * Batch size
    * Regularization strength
    * Momentum
    * ...

调参调的就是Hyperparameters

- 调参的方法：

  - Manual search:

    - experience-based

  - Grid search (structured, for ‘real’ applications)

    **代码见4.5.1**
  
    - Define ranges for all parameters spaces and select points
    - Usually pseudo-uniformly distributed

    Iterate over all possible configurations

    - 比如2个超参数要调，row横轴是第一个参数，column竖轴是第二个参数。这样每一个点代表一个可能性，迭代每一个点就是迭代了所有的可能性
  
  - Random search:
    
    **代码见4.5.2**
    
    - Like grid search but one picks points at random in the predefined ranges
    - To get a deeper understanding of random search and why it is more efficient than grid search, you should definitely check out this [paper](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf):
    
  - Auto-ML search
    - Bayesian framework; gradient descent on gradient descent, typically complex
  
  

## 1）损失函数Loss Functions

**代码实现**参见2.4

- Loss Function: 

  Used to measure the goodness of the predictions(the network's performance)

### 1.1 Classification loss:

  - **Cross-Entropy loss** 交叉熵损失函数(Maximum Likelihood Estimate)

    也叫做Softmax Loss，用于多分类中表示预测和真实分布之间的距离有多远。

    $$ CE(\hat{y}, y) = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^{C} \Big[ -y_{ik} \log(\hat{y}_{ik}) \Big]=-\frac{1}{N}\sum_{i=1}^Nlog(\frac{e^{S_{yi}}}{\Sigma_{k=1}^Ce^{S_k}}) $$

    $L_i=-log(\frac{e^{S_{}yi}}{\Sigma_{k=1}^Ce^{S_k}}) $

    where:
    - $ N $ is again the number of samples

    - $ C $ is the number of classes有多少个分类对象
      - 当C=2，即binary cross-entropy(BCE)
    - $ \hat{y}_{ik} $ is the probability that the model assigns for the $k$'th class when the $i$'th sample is the input. 
    - $y_{ik} = 1 $ iff the true label of the $i$th sample is $k$ and 0 otherwise. This is called a [one-hot encoding](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/).
    - $\frac{e^{S_{}yi}}{\Sigma_{k=1}^Ce^{S_k}} $:即softmax function,表示每一个类的概率
      - 分子是Exponential Operation：make sure probability > 0
      - 分母是Normalization正则化：make sure probabilities sum up to 1.
      - $s=f(x_i,\theta)$:Score function
    - 为什么叫Softmax Loss:因为Output Layer 是Softmax function

  -  **Binary Cross-Entropy** (BCE).用于二分类

    $$BCE(y,\hat{y}) =- y \cdot log(\hat y ) - (1- y) \cdot log(1-\hat y) $$

    - $y\in\mathbb{R}$ is the ground truth
    -  $\hat y\in\mathbb{R}$ is the predicted probability of the house being expensive.
    -  相比于多分类用的Softmax，BCE用的Output Layer是Sigmoid function
    
  - **Hinge Loss**:Multiclass SVM Loss

    $L=\frac{1}{N}\sum_{i=1}^NL_i$

    $L_i=\sum_{k\neq y_i}max(0,s_k-s_{y_i}+1)$

    - $s_k$：是不同class的分数
    - $s_{y_i}$：是真实class的分数

    理解：

    - 相比Cross Entropy loss永远不可能等于0，Hinge Loss会saturate(=0),2个不同的模型的loss值都等于0时，就无法分辨哪个更好

    -  实现了软间隔分类（这个Loss函数都可以做到）

    - **保持了支持向量机解的稀疏性**

      - **换用其他的Loss函数的话，SVM就不再是SVM了。**

        **正是因为HingeLoss的零区域对应的正是非支持向量的普通样本，从而所有的普通样本都不参与最终超平面的决定，这才是支持向量机最大的优势所在，对训练样本数目的依赖大大减少，而且提高了训练效率。**

### 1.2 Regression loss

  - **L1 loss**: $L(y,\hat{y};\theta)=\frac{1}{n}\sum_i^n||y_i-\hat y_i||_1$
    - Robust(Cost of outliers is linear)不容易受离异值影响
    - Costly to optimize
    - Optimum is the median
  - **L2/MSE loss**: $L(y,\hat{y};\theta)=\frac{1}{n}\sum_i^n||y_i-\hat{y}_i||_2^2$
    - Prone to outliers容易受离异值影响
    - Computer efficient optimization
    - Optimum is the mean

### 1.3 Loss + Regularization正则化

![](https://github.com/Fernweh-yang/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/Deep%20learning/Full%20Loss.png?raw=true)

- Full Loss = Data Loss + Reg Loss
  - Data Loss: 即选择用cross Entropy还是Hinge function来计算$L_i$,然后所有样本的loss值相加取平均

## 2) 激活函数Activation Functions

激活函数（Activation Function）类似于人类大脑中基于神经元的模型，激活函数最终决定了要发射给下一个神经元的内容。

- 激活函数为什么是非线性的
  - 如果使用线性激活函数，那么输入跟输出之间的关系为线性的，无论神经网络有多少层都是线性组合。
  - 激活函数给神经元引入了非线性因素，使得神经网络可以任意逼近任何非线性函数，这样神经网络就可以应用到众多的非线性模型中。
  - 输出层可能会使用线性激活函数，但在**隐含层都使用非线性激活函数**

### 2.1Sigmoid

![](https://github.com/Fernweh-yang/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/Deep%20learning/Sigmoid%20function.png?raw=true)

$\sigma(x)=\frac{1}{(1+e^{-x})}$

- Sigmoid is one of the oldest used non-linearities. 

  - 公式：$\sigma(x)=\frac{1}{1+exp(-x)}$


  - Sigmoid的导数：$\frac{\partial\sigma(x)}{\partial x}=\sigma(x)(1-\sigma(x))$


  - 反向求导：
    $$
    \frac{\partial L}{\partial x}=\frac{\partial L}{\partial \sigma(x)}\cdot\frac{\partial\sigma(x)}{\partial x}
    $$

    - $\frac{\partial L}{\partial \sigma(x)}$对应下面3.2.2代码backward中的`dout`
    
    - $\frac{\partial\sigma(x)}{\partial x}$对应下面3.2.2代码backward中的`sd`
    
  - 优点：
    - The output is between 0 and 1. So sigmoid can be used for output layer of classification-aimed neural networks

  - 缺点：
    - It requires **computation of an exponent**, which need more compute resouce and makes the convergence of the network slower.
    - **vanishing gradient**梯度消失：the neuro is saturated, because when it reaches its maximum or minimum value, the derivative will be equal to 0（也称为saturate饱和）.$\sigma(x)(1-\sigma(x))$趋于0，因为sigmoid最小最大值为0和1。then the weights will not be updated
    - **Not zero-centered function**:During update process, these weights are only allowed to move in one direction, i.e. positive or negative, at a time. This makes the loss function optimization harder.
### 2.2tanh

![](https://github.com/Fernweh-yang/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/Deep%20learning/Tanh%20function.png?raw=true)

- 公式：$tanh(x)=\frac{sinhx}{coshx}=\frac{e^x-e^{-x}}{e^x+e^{-x}}$
  -  tanh的导数：$\frac{\partial tanhx}{\partial x}=1-tan(x)^2$
- 优点：
  - The output is between -1 and 1. Unlike Sigmoid, It is a **zero-centered function** so that the optimization of the loss function becomes easier.
- 缺点：
  - Has the same problem as Sigmoid: **vanishing gradient** and **computation of an exponent**
- 代码实现见3.2.2

### 2.3ReLU

![](https://github.com/Fernweh-yang/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/Deep%20learning/ReLu%20function.png?raw=true)

$max(0,x)$

Rectified Linear Units线性整流单元 are the currently **most used** non-linearities in deep learning

- 公式:$ReLU(x) = max(0,x)=\begin{cases}x(x>0)\\0(x\leq0) \end{cases}$

  - ReLU的导数：$\frac{\partial ReLU(x)}{\partial x}=\begin{cases}1(x>0)\\0(x\leq0) \end{cases}$

- 反向求导:
  $$
  \frac{\partial L}{\partial x}=\frac{\partial L}{\partial ReLU(x)}\cdot\frac{\partial ReLU(x)}{\partial x}
  $$

  - $\frac{\partial L}{\partial ReLU(x)}$对应下面3.2.2代码backward中的dout
  - $\frac{\partial ReLU(x)}{\partial x}$如果>0值为1就是传dout本身，所以只需要让<=0的dout值为0即可。
  
- 优点：

  - It is easy to compute so that the neural network converges very quickly.
  - For positive values of the neuron: No vanishing gradient

- 缺点：

  - It is not a zero-centered function.

  - **Dead ReLU Problem(神经元坏死现象)**：For negative values, the gradient will be 0 forever and the weights weill not be updated anymore

    解决：

    1.  Use Xavier to initialize
    2. set a small learning rate

### 2.4Leaky ReLU

![](https://github.com/Fernweh-yang/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/Deep%20learning/Leaky%20ReLu%20function.png?raw=true)

 $当\alpha =0.01,即max(0.01x,x)$时，叫做leaky relu. 如果$\alpha$等于其他值，就叫randomized relu.

- 优点：
  - It is easy to compute.
  - It is close to zero-centered function.
  - Solved **Dead ReLU Problem(神经元坏死现象)**
- 代码实现见3.2.2

## 3) 优化算法Optimization 

**具体代码**实现见3.5

### 3.0 不同优化算法之间的关系

[参考1](https://zhuanlan.zhihu.com/p/81048198)

[参考2](https://ruder.io/optimizing-gradient-descent/)（老师推荐）

每次梯度下降都遍历整个数据集会耗费大量计算能力，而mini-batch梯度下降法通过从数据集抽取小批量的数据进行小批度梯度下降解决了这一问题。使用mini-batch会产生下降过程中左右振荡的现象。而动量梯度下降法通过减小振荡对算法进行优化。动量梯度下降法的核心便是对一系列梯度进行指数加权平均。

- Linear Systems(AX=b)
  - LU, QR, Cholesky, Jacobi, Gauss-Seidel, CG, PCG, etc.
- Non-linear (gradient-based)
  - [Newton, Gauss-Newton, LM, (L)BFGS](https://zhuanlan.zhihu.com/p/29672873)         <---Second order
  - Gradient Descent, SGD                                  <---first order
- Others
  - Genetic algorithms, MCMC, Metropolis-Hastings, etc.
  - Constrained and convex solvers (Langrage, ADMM,
    Primal-Dual, etc.)

### 3.1Gradient Descent

- Single Training Sample
  1. Given a loss function $L$ and a single training sample $\{x_i,y_i\}$
  2. Find best model parameters$\theta=\{W,b\}$
  3. Cost$L_i(\theta,x_i,y_i)$
     - $\theta=arg minL_i(x_i,y_i)$
  4. Gradient Descent
     1. Initialize$\theta^1$with random values
     2. Update step:$\theta^{k+1}=\theta^{k}-\alpha\nabla_\theta L_i(\theta^k,x_i,y_i)$
        - $\nabla_\theta L_i(\theta^k,x_i,y_i)$ computed via backpropagation
     3. Iterate until convergence:$|\theta^{k+1}-\theta^k|<\epsilon$

- Multiple Training Samples
  1. Given a loss function $L$ and multiple(n) training samples $\{x_i,y_i\}$
  2. Find best model parameters$\theta=\{W,b\}$
  3. Cost$L=\frac{1}{n}\sum_{i=1}^n L_i(\theta,x_i,y_i)$
     - $\theta=arg min L$
  4. Gradient Descent
     1. Initialize$\theta^1$with random values
     2. Update step:$\theta^{k+1}=\theta^{k}-\alpha\nabla_\theta L(\theta^k,x_{\{1..n\}},y_{\{1..n\}})$
        - Gradient is sum over residuals$\nabla_\theta L(\theta^k,x_{\{1..n\}},y_{\{1..n\}})=\frac{1}{n}\sum_{i=1}^n \nabla_\theta L_i(\theta,x_i,y_i)$
        - $\nabla_\theta L_i(\theta^k,x_i,y_i)$ computed via backpropagation
        - omitting省略$\frac{1}{n}$is not wrong: 可以看作resacling缩放 the learning rate

### 3.2Stochastic Gradient Descent(SGD)

随机梯度下降法-》小批量梯度下降法

如果有n=1million个training samples$\{x_i,y_i\}$,有50万个参数$\theta$，计算变得非常expensive无法一次性计算所有的samples。

根据empirical risk minimization经验风险最小化，可以用一小撮数据mini batch来近似计算。事实发现，实际训练中使用mini-batch梯度下降法可以大大加快训练速度，并节省计算机资源

实现方法：将样本总体分成多个mini-batch。例如100万的数据，分成10000份,每份包含100个数据的mini-batch-1到mini-batch-10000，每次梯度下降使用其中一个mini-batch进行训练，除此之外和梯度下降法没有任何区别
$$
L=\frac{1}{n}\sum_{i=1}^n L_i(\theta,x_i,y_i)\approx\frac{1}{|S|}\sum_{j\in S} L_j(\theta,x_j,y_j)
$$

- 与GD的对比

  ![](https://github.com/Fernweh-yang/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/Deep%20learning/%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D.jpg?raw=true)

  - 可以看到由于mini-batch每次仅使用数据集中的一部分进行梯度下降，所以每次下降并不是严格按照朝最小方向下降，但是总体下降趋势是朝着最小方向
  - 当然实际上GD也是曲折的，SGD会加剧曲折

- Minibatch: choose subset of transet $m<<n$
  $$
  \begin{aligned}
  B_i &= \{ \{x_1,y_1\},\{x_2,y_2\},..,\{x_m,y_m\}\} \\
    &= \{B_1,B_2,...B_{\frac{n}{m}}\}
  \end{aligned}
  $$

  - m: Minibatch size选择选择为 power of 2 即8，16，32.....
  - 越小的minibatch size ->  greater variance in the gradients -> noisy updates
  - It is limited by GPU memory(in backward pass)

- Stochastic Gradient Descent

  - $\theta^{k+1}=\theta^{k}-\alpha\nabla_\theta L(\theta^k,x_{\{1..m\}},y_{\{1..m\}})$
    - k现在refers to k-th **iteration**,$k\in(1.\frac{n}{m})$
    - 当所有的$\frac{n}{m}$个minibatch都更新完，就称为完成了一个**epoch**
  - $\nabla_\theta L=\frac{1}{m}\sum_{i=1}^m\nabla_\theta L_i$
    - m training samples in the current minibatch
    - $\nabla_\theta$: Gradient for the k-th minibatch

- Problems of SGD

  - Gradient is scaled equally均等缩放 across all dimensions
    - i.e., cannot independently scale directions
    - need to have conservative保守的 min learning rate to avoid divergence发散
    - Slower than ‘necessary’
  - Finding good learning rate is an art by itself

### 3.3Gradient Descent with Momentum

[动量梯度下降法](https://zhuanlan.zhihu.com/p/34240246)， 主要是针对mini-batch梯度下降法进行优化。

原理：给梯度下降加一个动量，让它在伪最优解的地方能够像现实生活中的小球那样冲出去。

- 好处：
  - 可以很轻松的跳出下面这三种伪最优解
    - plateau(稳定的水平)：下降的很慢，微分约等于0
    - saddle point(鞍点)：微分时沿着某一方向是稳定的(=0)，另一条方向是不稳定的奇点
    - local minima(局部最小点)；
  - 让寻找最优解的曲线能够不那么振荡、波动，让他能够更加的平滑，在水平方向的速度更快。（如下图）

- 与SGD的对比

  ![](https://github.com/Fernweh-yang/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/Deep%20learning/sgd_momentum.png?raw=true)

  - 可以看到动量优化之后左右的摆动减小，从而提高了效率。

-  Gradient Descent with Momentum

  - $v^{k+1}=\beta \cdot v^k-\alpha\cdot\nabla_\theta L(\theta^k)$
    - $v$: velocity。这是一个向量
    - $\beta$: accmulation rate积累速率(friction, momentum)，和$\alpha$一样都是Hyperparameters
  - $\theta^{k+1}=\theta^k+v^{k+1}$

### 3.4Root Mean Squared Prop(RMSProp)

均方根传递

Momentum虽然初步减小了摆动幅度但是实际应用中摆动幅度仍然较大和收敛速度较慢。RMSProp在动量法的基础上进一步优化该问题。具体做法是更新权重时，使用除根号的方法，使得较大幅度大幅度变小，较小幅度小幅度变小；然后通过设置较大learning rate，使得学习步子变大。

- 直观展示：

  ![](https://github.com/Fernweh-yang/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/Deep%20learning/RMSProp.png?raw=true)

  - Division in Y-Direction will be large
  - Division in X-Direction will be small

- RMSProp

  - $s^{k+1}=\beta\cdot s^k + (1-\beta)[\nabla_\theta L\bigodot \nabla_\theta L]$

    - $\nabla_\theta L\bigodot \nabla_\theta L$: 表示element wise multiplication

  - $\theta^{k+1}=\theta^k-\alpha\cdot\frac{\nabla_\theta L}{\sqrt{s^{k+1}+\epsilon}}$

    - $\epsilon,\alpha,\beta$都是Hyperparameters

- 特点：
  
  - Dampening抑制 the oscillations振荡 for high-variance高方差 directions
  - Can use faster learning rate because it is less likely to diverge
    - Speed up learning speed
    - Second moment二阶矩估计
  

### 3.5Adaptive Moment Estimation (Adam)

自适应矩估计，本质上是带有动量项momentum的RMSprop.它利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。Adam的优点主要在于经过偏置校正后，每一次迭代学习率都有个确定范围，使得参数比较平稳。

- Adam
  - 原理即结合Momentum和RMSProp
    - Momentum:   $m^{k+1}=\beta_1\cdot m^k+(1-\beta_1)\nabla_\theta L(\theta^k)$
    - RMSProp:        $v^{k+1}=\beta_2\cdot v^k + (1-\beta_2)[\nabla_\theta L\bigodot \nabla_\theta L]$
  - 初始化$m^0=0,v^0=0$
  - 跟新两个值
    - Momentum:$\hat{m}^{k+1}=\frac{m^{k+1}}{1-\beta_1^{k+1}}$
    - RMSProp:$\hat{v}^{k+1}=\frac{v^{k+1}}{1-\beta_2^{k+1}}$
  - 更新参数
    - $\theta^{k+1}=\theta^k-\alpha\cdot\frac{\hat{m}^{k+1}}{\sqrt{\hat{v}^{k+1}}+\epsilon}$
- 特点：
  - Exponentially-decaying指数衰减 mean and variance of gradients (combines first and second order momentum)
  - 优点：
    - 实现简单，计算高效，对内存需求少
    - 参数的更新不受梯度的伸缩变换影响
    - 超参数具有很好的解释性，且通常无需调整或仅需很少的微调
    - 更新的步长能够被限制在大致的范围内（初始学习率）
    - 能自然地实现步长退火过程（自动调整学习率）
    - 很适合应用于大规模的数据及参数的场景
    - 适用于不稳定目标函数
    - 适用于梯度稀疏或梯度存在很大噪声的问题
  - 在实际应用中 ，Adam为最常用的方法

### 3.6梯度下降之外的优化算法：

1. Newton’s Method牛顿法
   - BFGS and L-BFGS，Gauss-Newton，Levenberg，Levenberg-Marquardt
   - 牛顿法doesn’t work well for minibatches，适用于全数据集处理
2. Conjugate Gradien共轭梯度法
3. coordinate descent坐标下降法



## 4) Learning Rate

- 不同learning rate的implication可能后果

  ![](https://github.com/Fernweh-yang/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/Deep%20learning/learning%20rate.png?raw=true)

  - need high learning rate when far away aim
  - need low learning rate when close

- Learning Rate Decay衰败

  不同的Decay方法

  - $\alpha=\frac{1}{1+decay\_rate*epoch}\cdot\alpha_0$

  - step decay:$\alpha=\alpha-t\cdot\alpha$
    - only every n steps
    - t is decay rate(often 0.5)
  - Exponential decay:$\alpha=t^{epoch}\cdot\alpha_0$
    - t is decay rate<1.0
  - $\alpha =\frac{t}{\sqrt{epoch}}\cdot\alpha_0$

## 5) Performance measure

[参考](https://zhuanlan.zhihu.com/p/78204581)

- 标准：

  | Predicted class   \   Actual class | 1                | 0                |
  | ---------------------------------- | ---------------- | ---------------- |
  | 1                                  | True & Positive  | False & Positive |
  | 0                                  | False & Negative | True & Negative  |

  - $精准度precision=\frac{True\ Positive}{True\ Positive+False\ Positive}$

    - Precision从预测结果角度出发，描述了二分类器预测出来的正例结果中有多少是真实正例，即该二分类器预测的正例有多少是准确的

  - $召回率Recall = \frac{True\ Positive}{True\ Positive+False\ Negative}$

    - Recall从真实结果角度出发，描述了测试集中的真实正例有多少被二分类器挑选了出来，即真实的正例有多少被该二分类器召回。

  - $准确性Accuracy=\frac{True\ Positives + True\ Negatives}{Total\ Examples}$

    On **skewed使不公允 datasets**(e.g., when there are more positive examples than negative examples), accuracy is not a good measure of performance and you should instead use F1 score

  - $F_1\ Score=\frac{2 * Precision * Recall}{Precision + Recall}$

- Precision和Recall通常是一对矛盾的性能度量指标。一般来说，Precision越高时，Recall往往越低

## 6) 参数的初始化

- 初始化时权重Weights不能设置为如下三种
  1. 都等于0 ：The hidden units are all going to compute the same function, gradients are going to be the same
  2. 都是小随机数：Output become to zero，梯度消失vanishing gradient
  3. 都是大随机数：Output saturated to -1 and 1，对于tanh,sigmoid同样会梯度消失

### 6.1 Xavier Initialization

- Xavier初始化的作者，Xavier Glorot，在[论文](https://links.jianshu.com/go?to=http%3A%2F%2Fproceedings.mlr.press%2Fv9%2Fglorot10a%2Fglorot10a.pdf)中提出一个洞见：激活值的方差是逐层递减的，这导致反向传播中的梯度也逐层递减。要解决梯度消失，就要避免激活值方差的衰减，最理想的情况是，每层的输出值（激活值）保持高斯分布。

- 数学背景

  - 期望Expectation：是度量一个随机变量取值的集中位置或平均水平的最基本的数字特征

  - 方差Variance：是表示随机变量取值的分散性的一个数字特征

    -  方差越大，说明随机变量的取值分布越不均匀，变化性越强
    - 方差越小，说明随机变量的取值越趋近于均值，即期望值。

  - 关系有：

    - $E[X^2]=Var[X]+E[X]^2$

    如果X,Y是独立的，那么：

    - $Var[XY]=E[X^2Y^2]-E[XY]^2$
    - $E[XY]=E[X]E[Y]$

- Xavier Initialization
  $$
  \begin{align}
  Var(s)&=Var(\sum_i^nw_ix_i)=\sum_i^nVar(w_ix_i)\\
  &=\sum_i^n[E(w_i)]^2Var(x_i)+E[(x_i)]^2Var(w_i)+Var(x_i)Var(w_i)\\
  &=\sum_i^nVar(x_i)Var(w_i)\\
  &=n(Var(w)Var(x))
  \end{align}
  $$

  - $n$：the number of input neurons for the layer of weights you want to initialized

    注意n不是input Data$X\in R^{N\times D}$的N，对于第一层而言n=D

  - 上面等式第二行$E(w_i)和E(x_i)$因为用到Gaussian with zero mean所以都是0

  - 为了确保variance方差 of the output = input，即为了$Var(s)=Var(x)$

    - $n(Var(w)Var(x))=Var(x)\Rightarrow Var(w)=\frac{1}{n}$


### 6.2 Kaiming initialization

- 相比Xavier:
  - Xavier初始化的问题在于，它只适用于线性激活函数，但实际上，对于深层神经网络来说，线性激活函数是没有价值，神经网络需要非线性激活函数来构建复杂的非线性系统。今天的神经网络普遍使用relu激活函数。
  - Kaiming初始化的发明人kaiming he，在[论文](https://arxiv.org/abs/1502.01852)中提出了针对relu的kaiming初始化。
  - 因为relu会抛弃掉小于0的值，对于一个均值为0的data来说，这就相当于砍掉了一半的值，这样一来，均值就会变大，前面Xavier初始化公式中E(x)=mean=0的情况就不成立了。根据新公式的推导，最终得到新的rescale系数：$\sqrt\frac{2}{n}$
- $Var(w)=\frac{2}{n}$
  - $n$：the number of input neurons for the layer of weights you want to initialize

## 7) 数据集

- Training set('train')

  -  used for training neural network and get weights用于训练模型以及确定模型权重。

- Validation set('val')

  - Hyperparameter Optimization调整模型的超参数。
  - tune the model's architecture确定网络结构

- Test set('test')

  -  assess the performance [generalization]检验模型的泛化能力。

- Typocal split分割

  - train:60    val:20    test:20
  - train:80    val:10    test:10
  - train:98   val:1    test:1
  - train:80(cross-validation)    test:20

- [Cross-Validation](https://zhuanlan.zhihu.com/p/24825503)

  1. LOOCV(Leave-one-out cross-validation)

     只用一个数据作为测试集，其他的数据都作为训练集，并将此步骤重复N次（N为数据集的数据数量）。

     结果就是我们最终训练了n个模型，每次都能得到一个MSE（Mean Squarded Error）。而计算最终test MSE则就是将这n个MSE取平均

  2. K-fold cross validation k折交叉验证

     和LOOCV的不同在于，我们每次的测试集将不再只包含一个数据，而是多个，具体数目将根据K的选取决定。

     比如K=5，就把数据集分成5份。不重复地每次取其中一份做测试集，用其他四份做训练集训练模型，之后计算该模型在测试集上的MSE

- 各个数据集误差处理

  ![](https://github.com/Fernweh-yang/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/Deep%20learning/%E5%90%84%E6%95%B0%E6%8D%AE%E9%9B%86%E8%AF%AF%E5%B7%AE.png?raw=true)

  - Ideal Training： Small gap between training and validation loss, and both
    go down at same rate (stable without fluctuations波动).

## 8) Data Augmentation

- Data Pre-Processing

  ![](https://github.com/Fernweh-yang/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/Deep%20learning/Data%20pre_processing.png?raw=true)

- Data Augmentation增长
  - A classifier has to be invariant不变 to a wide variety of transformations变化
  - Helping the classifier: synthesize合成 data simulating plausible似乎是真的 transformations
  - 手段有：
    - Flip翻转
    - crop裁剪
    - Random brightniess and contrast changes
    - shear剪切(使倾斜)

## 9) Regularization 正则化

正则化的目的主要是防止过拟合overfitting，限制模型的复杂度。

### 9.1 L2 regularization

- 公式：$\theta_{k+1}=\theta_k-\epsilon\nabla_\theta(\theta_k,x,y)-\lambda\theta_k$

  - $\epsilon$: Learning rate

  - $\nabla_\theta$: Gradient

  - $-\lambda\theta_k$: Gradient of L2-Regularization
  - $1-\lambda$: Learning rate of weight decay衰减.(上面公式合并一下)

- 用于penalize惩罚 large weights

- 用于improve generalization泛化

- Early Stopping

### 9.2 Max Norm Regularization

Regularization technique that constrains weights of network (directly) 

Hyperparameter:$r\in\mathbb{R}>0$
$$
w=\begin{cases}
w \quad \quad if\ ||w||_2\leq r\\
r\frac{w}{||w||_2} \ otherwise
\end{cases}\\
||w||_2=\sqrt{w_1^2+w_2^2+\cdots+w_n^2}
$$

### 9.3 Ensemble Methods集成学习

  集成学习是一种训练思路，并不是某种具体的算法。它会挑选一些简单的基础模型进行组装，从而得到更好的效果。主要有2种：

  - **Bagging**(bootstrap aggregating自助聚集的简写)：核心思路是民主
    - Bagging 的思路是所有基础模型都一致对待，每个基础模型手里都只有一票。然后使用民主投票的方式得到最终的结果。
    - **经过 bagging 得到的结果方差（variance）更小**。
    - 具体过程：
      1. 从原始样本集中抽取训练集。每轮从原始样本集中使用Bootstraping的方法抽取n个训练样本（在训练集中，有些样本可能被多次抽取到，而有些样本可能一次都没有被抽中）。共进行k轮抽取，得到k个训练集。（k个训练集之间是相互独立的）
      2. 每次使用一个训练集得到一个模型，k个训练集共得到k个模型。（注：这里并没有具体的分类算法或回归方法，我们可以根据具体问题采用不同的分类或回归方法，如决策树、感知器等）
      3. 对分类问题：将上步得到的k个模型采用投票的方式得到分类结果；对回归问题，计算上述模型的均值作为最后的结果。（所有模型的重要性相同）
    - 在 bagging 的方法中，最广为熟知的就是随机森林了：bagging + 决策树 = 随机森林
  - **Boosting**助推：核心思路是精英
    - Boosting 和 bagging 最本质的差别在于他对基础模型不是一致对待的，而是经过不停的考验和筛选来挑选出「精英」，然后给精英更多的投票权，表现不好的基础模型则给较少的投票权，然后综合所有人的投票得到最终结果。
    - **经过 boosting 得到的结果偏差（bias）更小**
    - 具体过程：
      1. 通过加法模型将基础模型进行线性的组合。
      2. 每一轮训练都提升那些错误率小的基础模型权重，同时减小错误率高的模型权重。
      3. 在每一轮改变训练数据的权值或概率分布，通过提高那些在前一轮被弱分类器分错样例的权值，减小前一轮分对样例的权值，来使得分类器对误分的数据有较好的效果。
    - 在 boosting 的方法中，比较主流的有 [Adaboost](https://easyai.tech/ai-definition/adaboost/) 和 Gradient boosting 。

### 9.4 Dropout

- 在机器学习的模型中，如果模型的参数太多，而训练样本又太少，训练出来的模型很容易产生过拟合的现象。Dropout可以比较有效的缓解过拟合的发生，在一定程度上达到正则化的效果。

- Dropout可以作为训练深度神经网络的一种trick供选择。在每个训练批次中，通过忽略一半的特征检测器（让一半的隐层节点值为0）Using half the network = half capacity，可以明显地减少过拟合现象。这种方式可以减少特征检测器（隐层节点）间的相互作用reducing co-adaptation between neurons，检测器相互作用是指某些检测器依赖其他检测器才能发挥作用。

  ![](https://github.com/Fernweh-yang/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/Deep%20learning/Dropout.png?raw=true)

- 特征
  - Usually does not work well when combined with batch-norm.
  - Efficient regularization method, can be used with L2
  - Training takes a bit longer, usually 1.5x
  - But, can be used for uncertainty estimation
- Monte Carlo dropout
  - Neural networks are massively overconfident
  - We can use dropout to make the softmax probabilities more calibrated校准的.
  - Training: use dropout with a low p (0.1 or 0.2).
  - Inference, run the same image multiple times (25- 100), and average the results.

### 9.5 Batch normalization

- Batch Normalization批标准化：

  - Batch normalization 也可以被看做一个层面.用在fully connected or convolutional layers和 activation function之间。

    先数据X -> 全连接层 -> Batch normalization -> 激励函数

  - Batch normalization 的 batch 是批数据, 把数据分成小批小批进行 stochastic gradient descent. 而且在每批数据进行前向传递 forward propagation 的时候, 对每一层都进行 normalization 的处理,

- 为什么需要Batch Normalization批标准化：

  1. We know that normalizing input features can speed up learning, one intuition is that doing same thing for hidden layers should also work.
  2. solve the problem of covariance shift样本点的变化
     - Covariance shift: 假设x是属于特征空间的某一样本点，y是标签。covariate这个词，其实就是指这里的x。Covariance shift即样本点x的变化。
     - Suppose you have trained your cat-recognizing network use black cat, but evaluate on colored cats, you will see data distribution changing(called covariance shift). 我们无法用只用黑猫训练的模型去预测五颜六色的猫，这时候就只能重新训练模型
     - For a neural network, suppose input distribution is constant恒定, so output distribution of a certain hidden layer某个隐藏层的输出分布 should have been constant. But as the weights of that layer and previous layers changing in the training phase, the output distribution will change, this cause covariance shift from the perspective of layer after it从后面层的角度来看. Just like cat-recognizing network, the following need to re-train. To recover this problem, we use batch normal to force a zero-mean and one-variance distribution. It allow layer after it to learn independently from previous layers, and more **concentrate专注于 on its own task**, and so as to speed up the training process.
  3. Batch normal as regularization(slightly)
     - In batch normal, mean and variance is computed on mini-batch, which consist not too much samples. So the mean and variance contains noise. Just like dropout, it adds some noise to hidden layer's activation(dropout randomly multiply activation by 0 or 1). **This is an extra and slight effect, don't rely on it as a regularizer.**
  4. 对于tanh激活函数，可以avoid vanishing gradient
     - 没有 normalize 的数据 使用 tanh 激活以后, 激活值大部分都分布到了饱和阶段, 也就是大部分的激活值不是-1, 就是1, 而 normalize 以后, 大部分的激活值在每个分布区间都还有存在. 再将这个激活后的分布传递到下一层神经网络进行后续计算, 每个区间都有分布的这一种对于神经网络就会更加有价值.

  

- 公式：

  ![](https://github.com/Fernweh-yang/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/Deep%20learning/batch%20normalization.png?raw=true)

  - 用N个有D个特征的样本数据x,来计算平均值mean: $\mu$

  - 基于平均值$\mu$,计算方差variance: $\sigma$

  - 标准化的normalized input data: $\hat{x}$
    $$
    \hat{x}=\frac{x_{i,j}-\mu_j}{\sqrt{\sigma_j^2+\epsilon}}
    $$

    - $\hat{x}$也叫做unit gaussian
    - 计算得到的是一个mini batch的$\hat{x}$

  - 最后得到输出: $y_{i,j}=\gamma_j\hat{x}_{i,j}+\beta_j$

    - $\gamma和\beta$：是我们要去训练的参数

- Train vs Test

  - Mean 和 Variace的获得：
    - Train：通过mini-batch获得
    - Test：没有足够的测试数据集，所以 by running an exponentially weighted averaged across training minibatches.

## 10) Transfer Learning迁移学习

- [迁移学习](https://www.zhihu.com/question/41979241)(Transfer learning) 
  - 就是把已学训练好的模型参数迁移到新的模型来帮助新模型训练。考虑到大部分数据或任务是存在相关性的，所以通过迁移学习我们可以将已经学到的模型参数（也可理解为模型学到的知识）通过某种方式来分享给新模型从而加快并优化模型的学习效率不用像大多数网络那样从零学习（starting from scratch，tabula rasa）。
- Pytorch关于Transfer Learning的[教程](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

## 11）一些训练技巧

1. Debug:
   1. Use train/validation/test curves
      - Evaluation needs to be consistent
      - Numbers need to be comparable
   2. Only make one change at a time
2. start with a simple network.
3. Start with single training sample->increase samples
4. Did not overfit to single batch first
5. Don't forgot to toggle train/eval mode for network
6. Don't Forgot to call `.zero_grad()` (in PyTorch) before calling `.backward()`
7. Don't pass softmaxed outputs to a loss function that expects raw logits (or vice-versa)
   - Logit value见3.4，是没有正则化的值
   - Softmax后的值是正则了的值，见3.4

## 12）整合成接口来调试

1. 将所有的东西写成一个接口`solver`,只要输入参数就能计算不同的模型。**代码见2.10**

2. `solver`调用的模型model，我们也要写成一个函数，来方便调用调参。**代码见4.3**

# *.CNN卷积神经网络

Convolutional Neural Networks

- 为什么处理图片时要用卷积层来代替fc全连接层

  比如一个5X5的图片，3个RGB通道，那么一个神经元就需要75个权重Weights。如果1个全连接层有1000个神经元，那1层就要算75000个权重！！！更何况图片不可能只有5X5那么小。所以用全连接层来训练图片是impractical的

- Convolutions vs Fully-Connected

  - In contrast to fully-connected layers, we want to restrict限制 the degrees of freedom
    - FC is somewhat brute force
    - Convolutions are structured
  - Sliding window to with the same filter parameters to extract提取 image features
    - Concept观念 of weight sharing
    - Extract same features independent of location
  - Output size:
    - Fully-Connected layer: One layer of neurons, independent
    - Convolutional Layer: Neurons arranged排列 in 3 dimensions

## Image Filter

每一个核给成出一个不一样的图片过滤器

- Edge detection:
  $$
  \left[
   \begin{matrix}
     -1 & -1 & -1 \\
     -1 & 8 & -1 \\
     -1 & -1 & -1
    \end{matrix}
    \right]
  $$

- Box Mean
  $$
  \frac{1}{9}\left[
   \begin{matrix}
     1 & 1 & 1 \\
     1 & 1 & 1 \\
     1 & 1 & 1
    \end{matrix}
    \right]
  $$

- Sharpen
  $$
  \left[
   \begin{matrix}
     0 & -1 & 0 \\
     -1 & 5 & -1 \\
     0 & -1 & 0
    \end{matrix}
    \right]
  $$

- Gaussian blur
  $$
  \frac{1}{16}\left[
   \begin{matrix}
     1 & 2 & 1 \\
     2 & 4 & 2 \\
     1 & 2 & 1
    \end{matrix}
    \right]
  $$

## Convolution Layers: 

- Padding填充(图片四周填充一圈值)

  - Why padding
    - Sizes get small too quickly
    - corner pixel is only used once
  - 最常用的是填充0

- Dimensions:

  N=7：图片是7X7X3   （X3是RGB通道）

  F=3：过滤器是3X3X3 （最后X3要和输入对应）

  S=1/2/3：stride过滤器前进的格子数

  - 如果输入时32X32X3，用5个5X5X3的过滤器，那么过滤后输出是28x28x5。(注意最后X5和过滤的个数一致)，接着如果对28x28x5还要过滤，就要用5x5x5的过滤器了。
  - padding的圈数应该是：$p=\frac{F-1}{2}$
  - Padding后再过滤后的维度是：$(\left[\frac{N+2P-F}{S}\right]+1)\times(\left[\frac{N+2P-F}{S}\right]+1)$

## Pooling Layer池化层

- 与卷积层对比：

  - Conv Layer = feature extraction
    - Computes a feature in a given region
  - Pooling Layer = feature selection
    - Picks the strongest activation in a region

- Dimension:

  - Input is a volume of size $W_{in}\times H_{in}\times D_{in}$   宽，高，深度 
  - 2个超参数
    - Spatial filter extent F
    - Stride S
  - Input is a volume of size $W_{out}\times H_{out}\times D_{out}$
    - $W_{out}=\frac{W_{in}-F}{S}+1$
    - $H_{out}=\frac{H_{in}-F}{S}+1$
    - $D_{out}=D_{in}$

- 类型：

  - Max Pooling

    ![](https://github.com/Fernweh-yang/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/Deep%20learning/Max%20Pooling.png?raw=true)

  - Average Pooling

    ![](https://github.com/Fernweh-yang/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/Deep%20learning/Average%20Pooling.png?raw=true)

## CNN Prototype

![](https://github.com/Fernweh-yang/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/Deep%20learning/CNN%20Prototype.png?raw=true)

- 可见最后会用一层Fully connected layer
  - 用卷积层得出的特征features来做最终的决定
  - 一般1或2层全连接层

## Receptive Field感受野

- 感受野是指特征图上的某个点能看到的输入图像的区域,即特征图上的点是由输入图像中感受野大小区域的计算得到的
- 神经元感受野的值越大表示其能接触到的原始图像范围就越大，也意味着它可能蕴含更为全局，语义层次更高的特征；相反，值越小则表示其所包含的特征越趋向局部和细节。因此**感受野的值可以用来大致判断每一层的抽象层次**

## Spatial Batch Normalization

一般的batch normalization见9.5。

- BatchNorm for convolutional NN = spatial batchnorm

  - Input size$(N,C,W,H)\rightarrow(N,C,H\times W)$
  - Compute minibatch mean and variance across N, W, H (i.e. we compute mean/var for each channel C))

- Other Normalizations

  ![](https://github.com/Fernweh-yang/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/Deep%20learning/Normalizations.png?raw=true)

## Dropout for convolutional layers

全连接层是随机不激活某一个unit,但卷积层是三维结构的，这么做并不能提升性能。

所以Spatial Dropout randomly sets entire feature maps to zero

# 1. 数据预处理

## 1.1加载库

```python
import os
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

%load_ext autoreload
%autoreload 2
%matplotlib inline

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
```

## 1.2数据集下载

```python
# Set up the output dataset folder
i2dl_exercises_path = os.path.dirname(os.path.abspath(os.getcwd()))
cifar_root = os.path.join(i2dl_exercises_path, "datasets", "cifar10")

# Init the dataset and display downloading information this one time
dataset = ImageFolderDataset(
    root=cifar_root,
    force_download=False,
    verbose=True
)
```

## 1.3数据可视化

```python
def load_image_as_numpy(image_path):
    return np.asarray(Image.open(image_path), dtype=float)

classes = [
    'plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
]
num_classes = len(classes)
samples_per_class = 7
for label, cls in enumerate(sorted(classes)):
    for i in range(samples_per_class):
        image_path = os.path.join(
            cifar_root,
            cls,
            str(i+1).zfill(4) + ".png"
        )  # e.g. cifar10/plane/0001.png
        image = np.asarray(Image.open(image_path))  # open image as numpy array
        plt_idx = i * num_classes + label + 1  # calculate plot location in the grid
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(image.astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)  # plot class names above columns
plt.show()
```

## 1.4 通用数据集类(硬盘加载)

- 数据集类封装了1.将数据从给定地址加载，2.返回一个装着预处理完的数据的字典，这2种功能。

- 这通常是进行一个新深度学习工程的第一步！
- 需要有2个魔法方法，来实现类似于序列和映射的类：
  - `__len__(self)`：由此可以调用`len(dataset)`来获得数据集中照片数量。
  - `__getitem__(self,index)`：由此可以调用`dataset[i]`来直接获得数据集中的第i个图片。

```python
class ImageFolderDataset(Dataset):
    """CIFAR-10 dataset class"""
    def __init__(self, *args,
                 transform=None,
                 download_url="https://i2dl.dvl.in.tum.de/downloads/cifar10.zip",
                 **kwargs):
        super().__init__(*args, 
                         download_url=download_url,
                         **kwargs)
        
        self.classes, self.class_to_idx = self._find_classes(self.root_path)
        self.images, self.labels = self.make_dataset(
            directory=self.root_path,
            class_to_idx=self.class_to_idx
        )
        self.transform = transform
        
	# @staticmethod用于修饰类中的方法,使其可以在不创建类实例的情况下调用方法
    # 静态方法不可以引用类中的属性或方法，其参数列表也不需要约定的默认参数self。
    @staticmethod
    def _find_classes(directory):
        """
        Finds the class folders in a dataset
        :param directory: root directory of the dataset
        :returns: (classes, class_to_idx), where
          - classes is the list of all classes found
          - class_to_idx is a dict that maps class to label
        """
        classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    @staticmethod
    def make_dataset(directory, class_to_idx):
        """
        Create the image dataset by preparaing a list of samples
        Images are sorted in an ascending order by class and file name
        :param directory: root directory of the dataset
        :param class_to_idx: A dict that maps classes to labels
        :returns: (images, labels) where:
            - images is a list containing paths to all images in the dataset, NOT the actual images
            - labels is a list containing one label per image
        """
        images, labels = [], []

        for target_class in sorted(class_to_idx.keys()):
            label = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            for root, _, fnames in sorted(os.walk(target_dir)):
                for fname in sorted(fnames):
                    if fname.endswith(".png"):
                        path = os.path.join(root, fname)
                        images.append(path)
                        labels.append(label)

        assert len(images) == len(labels)
        return images, labels
    
    def __len__(self):        
        length = None
        length = len(self.images)
        return length

    @staticmethod
    def load_image_as_numpy(image_path):
        """Load image from image_path as numpy array"""
        # np.asarray(): Convert the input to an array
        return np.asarray(Image.open(image_path), dtype=float)

    def __getitem__(self, index):
        data_dict = None
        if self.transform  == None:
            data_dict = {'image': ImageFolderDataset.load_image_as_numpy(self.images[index]),
                         'label': self.labels[index]}
        else:
            data_dict = {'image': self.transform(ImageFolderDataset.load_image_as_numpy(self.images[index])),
                         'label': self.labels[index]}
        return data_dict
```

## 1.5通用数据集类(内存加载)

- 1.4用硬盘加载会很慢，因为我们每次访问数据集元素时都会访问单个文件，然后将这些文件加载到内存中。
- 所以如果数据集的全部大小比较小，比如1，2个G，就可以将他们全部放入内存，来加快访问速度

```python
# 继承1.4的父类来写
class MemoryImageFolderDataset(ImageFolderDataset):
    def __init__(self, root, *args,
                 transform=None,
                 download_url="https://i2dl.dvl.in.tum.de/downloads/cifar10memory.zip",
                 **kwargs):
        # Fix the root directory automatically
        if not root.endswith('memory'):
            root += 'memory'

        super().__init__(
            root, *args, download_url=download_url, **kwargs)
        
        with open(os.path.join(
            self.root_path, 'cifar10.pckl'
            ), 'rb') as f:
            save_dict = pickle.load(f)

        self.images = save_dict['images']
        self.labels = save_dict['labels']
        self.class_to_idx = save_dict['class_to_idx']
        self.classes = save_dict['classes']

        self.transform = transform

    def load_image_as_numpy(self, image_path):
        """Here we already have everything in memory,
        so we can just return the image"""
        return image_path
    
    def __getitem__(self, index):
        data_dict = None

        if self.transform  == None:
            data_dict = {'image': MemoryImageFolderDataset.load_image_as_numpy(self,self.images[index]),
                         'label': self.labels[index]}
        else:
            data_dict = {'image': self.transform(MemoryImageFolderDataset.load_image_as_numpy(self,self.images[index])),
                         'label': self.labels[index]}

        return data_dict 
```

## 1.6 改变数据集尺寸大小

比如将图片的RGB参数从0-255改成0-1

```python
class RescaleTransform:
    """Transform class to rescale images to a given range"""
    def __init__(self, out_range=(0, 1), in_range=(0, 255)):
        """
        :param out_range: Value range to which images should be rescaled to
        :param in_range: Old value range of the images
            e.g. (0, 255) for images with raw pixel values
        """
        self.min = out_range[0]
        self.max = out_range[1]
        self._data_min = in_range[0]
        self._data_max = in_range[1]
	
    ##该方法使得类实例对象可以像调用普通函数那样，以“对象名()”的形式使用
    def __call__(self, images):
        # Formula:https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
        images = (self.max-self.min)*(images-self._data_min)/(self._data_max-self._data_min)+self.min

        return images
```

## 1.7 正则化数据(高斯)

先用numpy库计算数据集的mean平均值和standard deviation标准差

```python
def compute_image_mean_and_std(images):
    """
    Calculate the per-channel image mean and standard deviation of given images
    :param images: numpy array of shape NxHxWxC
        (for N images with C channels of spatial size HxW)
    :returns: per-channels mean and std; numpy array of shape C
    """
    mean, std = None, None
    mean = np.mean(images,axis=(0,1,2))
    std = np.std(images,axis=(0,1,2))

    return mean, std
```

再正则化数据

```python
class NormalizeTransform:
    """
    Transform class to normalize images using mean and std
    Functionality depends on the mean and std provided in __init__():
        - if mean and std are single values, normalize the entire image
        - if mean and std are numpy arrays of size C for C image channels,
            then normalize each image channel separately
    """
    def __init__(self, mean, std):
        """
        :param mean: mean of images to be normalized
            can be a single value, or a numpy array of size C
        :param std: standard deviation标准差 of images to be normalized
             can be a single value or a numpy array of size C
        """
        self.mean = mean    
        self.std = std
        
	##该方法使得类实例对象可以像调用普通函数那样，以“对象名()”的形式使用。
    # 1.实例化：normalize_transform = NormalizeTransform()
    # 2.调用： normalize_transform(images)
    def __call__(self, images):
        images = (images-self.mean)/self.std
        return images
```

## 1.8正则，改尺寸合二为一

只需要`transform=ComposeTransform([rescale_transform, normalize_transform])`就可以把1.6,1.7两种转换一次执行

```python
class ComposeTransform:
    """Transform class that combines multiple other transforms into one"""
    def __init__(self, transforms):
        """
        :param transforms: transforms to be combined
        """
        self.transforms = transforms

    def __call__(self, images):
        for transform in self.transforms:
            images = transform(images)
        return images
```

## 1.9实现加载数据并处理

```python
rescale_transform = RescaleTransform()
normalize_transform = NormalizeTransform(
    mean=cifar_mean,
    std=cifar_std
)

dataset = ImageFolderDataset(
    root=cifar_root,
    transform=ComposeTransform([rescale_transform, normalize_transform])
)
```

## 1.10批量加载数据

不应该一次性将所有的数据集都加载，而应该分批(batches)加载。

1. In machine learning, we often need to load data in **mini-batches**, which are small subsets of the training dataset. How many samples to load per mini-batch is called the **batch size**.
2. In addition to the Dataset class, we use a **DataLoader** class that takes care of mini-batch construction, data shuffling, and more.
3. The dataloader is iterable and only loads those samples of the dataset that are needed for the current mini-batch. This can lead to bottlenecks瓶颈 later if you are unable to provide enough batches in time for your upcoming pipeline. This is especially true when loading from HDDs as the slow reading time can be a bottleneck in your complete pipeline later.
4. The dataloader task can easily by distributed amongst multiple processes as well as pre-fetched. When we switch to PyTorch later we can directly use our dataset classes and replace our current Dataloader with theirs :).

```python
class DataLoader:
    """
    Dataloader Class
    Defines an iterable batch-sampler over a given dataset
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        """
        :param dataset: dataset from which to load the data
        :param batch_size: how many samples per batch to load
        :param shuffle: set to True to have the data reshuffled at every epoch
        :param drop_last: set to True to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size.
            If False and the size of dataset is not divisible by the batch
            size, then the last batch will be smaller.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
	
    ## 迭代器
    def __iter__(self):

        # 将batches整合成字典
        def combine_batch_dicts(batch):
            batch_dict = {}
            for data_dict in batch:
                for key,value in data_dict.items():
                    if key not in batch_dict:
                        batch_dict[key] = []
                    batch_dict[key].append(value)
            return batch_dict

        # 将字典种的value从原来的列表改成numpy的数组
        def batch_to_numpy(batch):
            numpy_batch = {}
            for key, value in batch.items():
                numpy_batch[key] = np.array(value)
            return numpy_batch

        # 将batches迭代器化
        if self.shuffle:
            index_iterator = iter(np.random.permutation(len(self.dataset)))  # define indices as iterator
        else:
            index_iterator = iter(range(len(self.dataset)))  # define indices as iterator

        batch = []
        for index in index_iterator:  # iterate over indices using the iterator
            batch.append(self.dataset[index])
            if len(batch) == self.batch_size:
                yield batch_to_numpy(combine_batch_dicts(batch))  # use yield keyword to define a iterable generator
                batch = []

        if self.drop_last == False and len(batch) > 0:
            yield batch_to_numpy(combine_batch_dicts(batch))

    def __len__(self):
        length = None
        temp = len(self.dataset)
        length = temp//self.batch_size
        if self.drop_last == False and temp/self.batch_size:
            length = temp//self.batch_size + 1 
        return length

```

# 2. 逻辑回归：判断房家贵/不贵

## 2.1加载库

```python
from exercise_code.data.csv_dataset import CSVDataset
from exercise_code.data.csv_dataset import FeatureSelectorAndNormalizationTransform
from exercise_code.data.dataloader import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns


pd.options.mode.chained_assignment = None  # default='warn'

%matplotlib inline
%load_ext autoreload
%autoreload 2
```

## 2.2数据集下载

```python
from exercise_code.networks.utils import *

X_train, y_train, X_val, y_val, X_test, y_test, train_dataset = get_housing_data()

print("train data shape:", X_train.shape)
print("train targets shape:", y_train.shape)
print("val data shape:", X_val.shape)
print("val targets shape:", y_val.shape)
print("test data shape:", X_test.shape)
print("test targets shape:", y_test.shape, '\n')

print('The original dataset looks as follows:')
train_dataset.df.head()
```

## 2.3分类模型

- 模型：$\mathbf{\hat{y}}  = \sigma \left( \mathbf{X} \cdot \mathbf{w} \right)$
  - $\mathbf{X} \in \mathbb{R}^{N\times (D+1)}$ be our data with $N$ samples and $D$ feature dimensions
  - $\mathbf{w}\in \mathbb{R}^{(D+1) \times 1}$ is the weight matrix of our model
  -  **sigmoid function**：$ \sigma(t) = \frac{1}{1+e^{-t}},\in[0.1] $

```python
    def sigmoid(self, x):
        """
        Computes the ouput of the sigmoid function

        :param x: input of the sigmoid, np.array of any shape
        :return: output of the sigmoid with same shape as input vector x
        """
        out = None

        out = 1/(1+np.exp(-x))

        return out
```



## 2.4损失函数

用于确定我们的模型是否足够好

```python

import os
import pickle
import numpy as np
from exercise_code.networks.linear_model import *


class Loss(object):
    def __init__(self):
        self.grad_history = []

    def forward(self, y_out, y_truth):
        return NotImplementedError

    def backward(self, y_out, y_truth, upstream_grad=1.):
        return NotImplementedError

    def __call__(self, y_out, y_truth):
        loss = self.forward(y_out, y_truth)
        grad = self.backward(y_out, y_truth)
        return (loss, grad)


class L1(Loss):

    def forward(self, y_out, y_truth):
        """
        Performs the forward pass of the L1 loss function.

        :param y_out: [N, ] array predicted value of your model.
               y_truth: [N, ] array ground truth value of your training set.
        :return: [N, ] array of L1 loss for each sample of your training set.
        """
        result = None
        result = np.abs(y_out - y_truth)
        return result

    def backward(self, y_out, y_truth):
        """
        Performs the backward pass of the L1 loss function.

        :param y_out: [N, ] array predicted value of your model.
               y_truth: [N, ] array ground truth value of your training set.
        :return: [N, ] array of L1 loss gradients w.r.t y_out for
                  each sample of your training set.
        """
        gradient = None
        gradient = y_out - y_truth

        zero_loc = np.where(gradient == 0)
        negative_loc = np.where(gradient < 0)
        positive_loc = np.where(gradient > 0)

        gradient[zero_loc] = 0
        gradient[positive_loc] = 1
        gradient[negative_loc] = -1

        return gradient


class MSE(Loss):

    def forward(self, y_out, y_truth):
        """
        Performs the forward pass of the MSE loss function.

        :param y_out: [N, ] array predicted value of your model.
                y_truth: [N, ] array ground truth value of your training set.
        :return: [N, ] array of MSE loss for each sample of your training set.
        """
        result = None
        result = (y_out - y_truth)**2
        return result

    def backward(self, y_out, y_truth):
        """
        Performs the backward pass of the MSE loss function.

        :param y_out: [N, ] array predicted value of your model.
               y_truth: [N, ] array ground truth value of your training set.
        :return: [N, ] array of MSE loss gradients w.r.t y_out for
                  each sample of your training set.
        """
        gradient = None
        gradient = 2 * (y_out - y_truth)
        return gradient


class BCE(Loss):

    def forward(self, y_out, y_truth):
        """
        Performs the forward pass of the binary cross entropy loss function.

        :param y_out: [N, ] array predicted value of your model.
                y_truth: [N, ] array ground truth value of your training set.
        :return: [N, ] array of binary cross entropy loss for each sample of your training set.
        """
        result = None

        result = -y_truth*np.log(y_out)-(1-y_truth)*np.log(1-y_out)


        return result

    def backward(self, y_out, y_truth):
        """
        Performs the backward pass of the loss function.

        :param y_out: [N, ] array predicted value of your model.
               y_truth: [N, ] array ground truth value of your training set.
        :return: [N, ] array of binary cross entropy loss gradients w.r.t y_out for
                  each sample of your training set.
        """
        gradient = None

        gradient = -y_truth/y_out +(1-y_truth)/(1-y_out)

        return gradient
    
class CrossEntropyFromLogits(Loss):
    def __init__(self):
        self.cache = {}
     
    def forward(self, y_out, y_truth, reduction='mean'):
        """
        Performs the forward pass of the cross entropy loss function.
        
        :param y_out: [N, C] array with the predicted logits of the model
            (i.e. the value before applying any activation)
        :param y_truth: [N, ] array with ground truth labels.
        
        :return: float, the cross-entropy loss value
        """
        
        # Transform the ground truth labels into one hot encodings.
        N, C = y_out.shape
        y_truth_one_hot = np.zeros_like(y_out)
        y_truth_one_hot[np.arange(N), y_truth] = 1
        
        # Transform the logits into a distribution using softmax.
        y_out_exp = np.exp(y_out - np.max(y_out, axis=1, keepdims=True))
        y_out_probs = y_out_exp / np.sum(y_out_exp, axis=1, keepdims=True)
        
        # Compute the loss for each element in the batch.
        loss = -y_truth_one_hot * np.log(y_out_probs)
        loss = loss.sum(axis=1).mean()
           
        self.cache['probs'] = y_out_probs
        
        return loss
    
    def backward(self, y_out, y_truth):
        N, C = y_out.shape
        gradient = self.cache['probs']
        gradient[np.arange(N), y_truth] -= 1
        gradient /= N
        
        return gradient
```

## 2.5前向传播，反向传播

- 前向传播用的2.3的模型：$\mathbf{\hat{y}}  = \sigma \left( \mathbf{X} \cdot \mathbf{w} \right)$
- 反向传播即它的导数：
  - $$\frac{\partial \hat y}{\partial w} = \frac{\partial \sigma(s)}{\partial w} = \frac{\partial \sigma(s)}{\partial s} \cdot \frac{\partial s}{\partial w}$$
  - $\frac{\partial \sigma(s)}{\partial s}=\sigma(s)*(1-\sigma(s))$
  - $\frac{\partial s}{\partial w}=X$

```python
   def forward(self, X):
        """
        Performs the forward pass of the model.

        :param X: N x D array of training data. Each row is a D-dimensional point.
        :return: Predicted labels for the data in X, shape N x 1
                 1-dimensional array of length N with classification scores.
        """
        assert self.W is not None, "weight matrix W is not initialized"
        # add a column of 1s to the data for the bias term
        batch_size, _ = X.shape
        X = np.concatenate((X, np.ones((batch_size, 1))), axis=1)
        # save the samples for the backward pass
        self.cache = X
        # output variable
        y = None

        y = self.sigmoid(np.dot(X, self.W))

        return y

    def backward(self, y):
        """
        Performs the backward pass of the model.

        :param y: N x 1 array. The output of the forward pass.
        :return: Gradient of the model output (y=sigma(X*W)) wrt W
        """
        assert self.cache is not None, "run a forward pass before the backward pass"
        dW = None


        ds = self.sigmoid(np.dot(self.cache, self.W))
        dW = self.cache * (ds * (1 - ds))


        return dW
```



## 2.6 优化器:梯度下降

- Optimizer:Gradient Descent 修改我们的参数使损失函数足够的小
  0. Initialize the weights with random values.
  1. Calculate loss with the current weights and the loss function.
  2. Calculate the gradient of the loss function w.r.t. the weights.
  3. Update weights with the corresponding gradient.
     - $ w^{(n+1)} = w^{(n)} - \alpha \cdot \frac {dL}{dw}, $
  4. Iteratively perform Step 1 to 3 until converges.

```python
class Optimizer(object):
    def __init__(self, model, learning_rate=5e-5):
        self.model = model
        self.lr = learning_rate

    def step(self, dw):
        """
        :param dw: [D+1,1] array gradient of loss w.r.t weights of your linear model
        :return weight: [D+1,1] updated weight after one step of gradient descent
        """
        weight = self.model.W

        weight = weight - self.lr*dw

        self.model.W = weight
```

## 2.7初始化权重

一个很low的随机方式来初始化权重

```python
    def initialize_weights(self, weights=None):
        """
        Initialize the weight matrix W

        :param weights: optional weights for initialization
        """
        if weights is not None:
            assert weights.shape == (self.num_features + 1, 1), \
                "weights for initialization are not in the correct shape (num_features + 1, 1)"
            self.W = weights
        else:
            self.W = 0.001 * np.random.randn(self.num_features + 1, 1)
```

## 2.8训练模型

1. 不经训练的模型

```python
from exercise_code.networks.classifier import Classifier

#initialization
model = Classifier(num_features=1)
model.initialize_weights()

y_out, _ = model(X_train)

# plot the prediction
plt.scatter(X_train, y_train)
plt.plot(X_train, y_out, color='r')
```

2. 训练后的模型更准确

```python
from exercise_code.networks.optimizer import *
from exercise_code.networks.classifier import *
# Hyperparameter Setting, we will specify the loss function we use, and implement the optimizer we finished in the last step.
num_features = 1

# initialization
model = Classifier(num_features=num_features)
model.initialize_weights()

loss_func = BCE() 
learning_rate = 5e-1
loss_history = []
opt = Optimizer(model,learning_rate)

steps = 400
# Full batch Gradient Descent
for i in range(steps):
    
    # Enable your model to store the gradient.
    model.train()
    
    # 因为Classifier的父类Networks和BCE的父类loss，都定义了__call__()这一魔法方法，所以才能像下面这样调用。
    # Compute the output and gradients w.r.t weights of your model for the input dataset.
    model_forward, model_backward = model(X_train)
    # 因为Classifier的父类Networks和BCE的父类loss，都定义了__call__()这一魔法方法，所以才能像下面这样调用。
    # Compute the loss and gradients w.r.t output of the model.
    loss, loss_grad = loss_func(model_forward, y_train)
    
    # Use back prop method to get the gradients of loss w.r.t the weights.
    grad = loss_grad * model_backward
    
    # Compute the average gradient over your batch
    grad = np.mean(grad, 0, keepdims = True)

    # After obtaining the gradients of loss with respect to the weights, we can use optimizer to
    # do gradient descent step.
    # Take transpose to have the same shape ([D+1,1]) as weights.
    opt.step(grad.T)
    
    # Average over the loss of the entire dataset and store it.
    average_loss = np.mean(loss)
    loss_history.append(average_loss)
    if i%10 == 0:
        print("Epoch ",i,"--- Average Loss: ", average_loss)

```

## 2.9可视化训练成果

```python
# Plot the loss history to see how it goes after several steps of gradient descent.
plt.plot(loss_history, label = 'Train Loss')
plt.xlabel('iteration')
plt.ylabel('training loss')
plt.title('Training Loss history')
plt.legend()
plt.show()


# forward pass
y_out, _ = model(X_train)


# plot the prediction
plt.scatter(X_train, y_train, label = 'Ground Truth')
inds = X_train.argsort(0).flatten()
plt.plot(X_train[inds], y_out[inds], color='r', label = 'Prediction')
plt.title('Prediction of our trained model')
plt.legend()
plt.show()
```

## 2.10一个通用的调用模型的方法

```python
import numpy as np
from exercise_code.networks.optimizer import Adam
from exercise_code.networks import CrossEntropyFromLogits


class Solver(object):
    """
    A Solver encapsulates all the logic necessary for training classification
    or regression models.
    The Solver performs gradient descent using the given learning rate.

    The solver accepts both training and validataion data and labels so it can
    periodically check classification accuracy on both training and validation
    data to watch out for overfitting.

    To train a model, you will first construct a Solver instance, passing the
    model, dataset, learning_rate to the constructor.
    You will then call the train() method to run the optimization
    procedure and train the model.

    After the train() method returns, model.params will contain the parameters
    that performed best on the validation set over the course of training.
    In addition, the instance variable solver.loss_history will contain a list
    of all losses encountered during training and the instance variables
    solver.train_loss_history and solver.val_loss_history will be lists
    containing the losses of the model on the training and validation set at
    each epoch.
    """

    def __init__(self, model, train_dataloader, val_dataloader,
                 loss_func=CrossEntropyFromLogits(), learning_rate=1e-3,
                 optimizer=Adam, verbose=True, print_every=1, lr_decay = 1.0,
                 **kwargs):
        """
        Construct a new Solver instance.

        Required arguments:
        - model: A model object conforming to the API described above

        - train_dataloader: A generator object returning training data
        - val_dataloader: A generator object returning validation data

        - loss_func: Loss function object.
        - learning_rate: Float, learning rate used for gradient descent.

        - optimizer: The optimizer specifying the update rule

        Optional arguments:
        - verbose: Boolean; if set to false then no output will be printed during
          training.
        - print_every: Integer; training losses will be printed every print_every
          iterations.
        """
        self.model = model
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.loss_func = loss_func

        self.opt = optimizer(model, loss_func, learning_rate)

        self.verbose = verbose
        self.print_every = print_every

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        self.current_patience = 0

        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.best_model_stats = None
        self.best_params = None

        self.train_loss_history = []
        self.val_loss_history = []

        self.train_batch_loss = []
        self.val_batch_loss = []

        self.num_operation = 0
        self.current_patience = 0

    def _step(self, X, y, validation=False):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.

        :param X: batch of training features
        :param y: batch of corresponding training labels
        :param validation: Boolean indicating whether this is a training or
            validation step

        :return loss: Loss between the model prediction for X and the target
            labels y
        """
        loss = None

        # Forward pass
        y_pred = self.model.forward(X)
        # Compute loss
        loss = self.loss_func.forward(y_pred, y)
        # Add the regularization
        loss += sum(self.model.reg.values())

        # Count number of operations
        self.num_operation += self.model.num_operation

        # Perform gradient update (only in train mode)
        if not validation:
            # Compute gradients
            self.opt.backward(y_pred, y)
            # Update weights
            self.opt.step()

            # If it was a training step, we need to count operations for
            # backpropagation as well
            self.num_operation += self.model.num_operation

        return loss

    def train(self, epochs=100, patience = None):
        """
        Run optimization to train the model.
        """

        # Start an epoch
        for t in range(epochs):

            # Iterate over all training samples
            train_epoch_loss = 0.0

            for batch in self.train_dataloader:
                # Unpack data
                X = batch['image']
                y = batch['label']

                # Update the model parameters.
                validate = t == 0
                train_loss = self._step(X, y, validation=validate)

                self.train_batch_loss.append(train_loss)
                train_epoch_loss += train_loss

            train_epoch_loss /= len(self.train_dataloader)

            
            self.opt.lr *= self.lr_decay
            
            
            # Iterate over all validation samples
            val_epoch_loss = 0.0

            for batch in self.val_dataloader:
                # Unpack data
                X = batch['image']
                y = batch['label']

                # Compute Loss - no param update at validation time!
                val_loss = self._step(X, y, validation=True)
                self.val_batch_loss.append(val_loss)
                val_epoch_loss += val_loss

            val_epoch_loss /= len(self.val_dataloader)

            # Record the losses for later inspection.
            self.train_loss_history.append(train_epoch_loss)
            self.val_loss_history.append(val_epoch_loss)

            if self.verbose and t % self.print_every == 0:
                print('(Epoch %d / %d) train loss: %f; val loss: %f' % (
                    t + 1, epochs, train_epoch_loss, val_epoch_loss))

            # Keep track of the best model
            self.update_best_loss(val_epoch_loss, train_epoch_loss)
            if patience and self.current_patience >= patience:
                print("Stopping early at epoch {}!".format(t))
                break

        # At the end of training swap the best params into the model
        self.model.params = self.best_params

    def get_dataset_accuracy(self, loader):
        correct = 0
        total = 0
        for batch in loader:
            X = batch['image']
            y = batch['label']
            y_pred = self.model.forward(X)
            label_pred = np.argmax(y_pred, axis=1)
            correct += sum(label_pred == y)
            if y.shape:
                total += y.shape[0]
            else:
                total += 1
        return correct / total

    def update_best_loss(self, val_loss, train_loss):
        # Update the model and best loss if we see improvements.
        if not self.best_model_stats or val_loss < self.best_model_stats["val_loss"]:
            self.best_model_stats = {"val_loss":val_loss, "train_loss":train_loss}
            self.best_params = self.model.params
            self.current_patience = 0
        else:
            self.current_patience += 1

```

- 调用这个类去运行我们的模型

  ```python
  #We use the BCE loss
  loss = BCE()
  
  # Please use these hyperparmeter as we also use them later in the evaluation
  learning_rate = 1e-1
  epochs = 25000
  
  # Setup for the actual solver that's going to do the job of training
  # the model on the given data. set 'verbose=True' to see real time 
  # progress of the training.
  solver = Solver(model, 
                  data, 
                  loss,
                  learning_rate, 
                  verbose=True, 
                  print_every = 1000)
  # Train the model, and look at the results.
  solver.train(epochs)
  
  
  # Test final performance
  y_out, _ = model(X_test)
  
  accuracy = test_accuracy(y_out, y_test)
  print("Accuracy AFTER training {:.1f}%".format(accuracy*100))
  ```


# 3. 数据集CIFAR10分类

## 3.1加载库

```
# As usual, a bit of setup
import matplotlib.pyplot as plt
import numpy as np
import os

from exercise_code.data import (
    DataLoader,
    ImageFolderDataset,
    RescaleTransform,
    NormalizeTransform,
    FlattenTransform,
    ComposeTransform,
)
from exercise_code.networks import (
    ClassificationNet,
    CrossEntropyFromLogits
)
from exercise_code.tests.layer_tests import *
from exercise_code.tests.sgdm_tests import *

from exercise_code.solver import Solver
from exercise_code.networks.optimizer import (
    SGD,
    SGDMomentum,
    Adam
)
from exercise_code.networks.compute_network_size import *

%load_ext autoreload
%autoreload 2
%matplotlib inline

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
```

## 3.2神经网络模型

### 3.2.1Modularization模块化

模块化chain rule:$\frac{\partial f}{\partial y}=\frac{\partial f}{\partial d}\cdot\frac{\partial d}{\partial y}$

- in the `forward` pass do all required computations as well as save all values that are required to compute gradients

  ```python
  def layer_forward(x, w):
    """ Receive inputs x and weights w """
    # Do some computations ...
    z = # ... some intermediate value
    # Do some more computations ...
    out = # the output
    cache = (x, w, z, out) # Values we need to compute gradients
    return out, cache
  ```

- in the `backward` function they will use the incoming gradients from later building blocks, to compute their respective各自的 gradients using their cached values.

  ```python
  def layer_backward(dout, cache):
    """
    Receive derivative of loss with respect to outputs and cache,
    and compute derivative with respect to inputs.
    """
    # Unpack cache values
    x, w, z, out = cache
    # Use values in cache to compute derivatives
    dx = # Derivative of loss with respect to x
    dw = # Derivative of loss with respect to w
    return dx, dw
  ```

- We can just simply chain an arbitrary amount of such blocks, so called `layers`

### 3.2.2Layer Non-Linearities(Sigmoid,Relu,Tanh,L-Relu)

1. **Sigmoid**

   Sigmoid is one of the oldest used non-linearities. 

   - 公式：$\sigma(x)=\frac{1}{1+exp(-x)}$

   - Sigmoid的导数：$\frac{\partial\sigma(x)}{\partial x}=\sigma(x)(1-\sigma(x))$

   - 反向求导：
     $$
     \frac{\partial L}{\partial x}=\frac{\partial L}{\partial \sigma(x)}\cdot\frac{\partial\sigma(x)}{\partial x}
     $$

     - $\frac{\partial L}{\partial \sigma(x)}$对应下面代码backward中的`dout`
     - $\frac{\partial\sigma(x)}{\partial x}$对应下面代码backward中的`sd`

   ```python
   class Sigmoid:
       def __init__(self):
           pass
   
       def forward(self, x):
           """
           :param x: Inputs, of any shape.
   
           :return out: Outputs, of the same shape as x.
           :return cache: Cache, stored for backward computation, of the same shape as x.
           """
           shape = x.shape
           out, cache = np.zeros(shape), np.zeros(shape)
   
           #out用于计算前向传播，cache用于计算反向传播
           out = 1/(1+np.exp(-x))
           cache = out
   
           return out, cache
   
       def backward(self, dout, cache):
           """
           :param dout: Upstream gradient from the computational graph, from the Loss function
                       and up to this layer. Has the shape of the output of forward().
           :param cache: The values that were stored during forward() to the memory,
                       to be used during the backpropogation.
           :return: dx: the gradient w.r.t. input X, of the same shape as X
           """
           dx = None
   
           sd = cache*(1-cache)
           dx = dout*sd
   
           return dx
   ```

2. **Relu**

   Rectified Linear Units线性整流单元 are the currently most used non-linearities in deep learning

   - 公式:$ReLU(x) = max(0,x)=\begin{cases}x(x>0)\\0(x\leq0) \end{cases}$

     - ReLU的导数：$\frac{\partial ReLU(x)}{\partial x}=\begin{cases}1(x>0)\\0(x\leq0) \end{cases}$

   - 反向求导:
     $$
     \frac{\partial L}{\partial x}=\frac{\partial L}{\partial ReLU(x)}\cdot\frac{\partial ReLU(x)}{\partial x}
     $$

     - $\frac{\partial L}{\partial ReLU(x)}$对应下面代码backward中的dout
     - $\frac{\partial ReLU(x)}{\partial x}$如果>0值为1就是传dout本身，所以只需要让<=0的dout值为0即可。

   ```python
   class Relu:
       def __init__(self):
           pass
   
       def forward(self, x):
           """
           :param x: Inputs, of any shape.
   
           :return outputs: Outputs, of the same shape as x.
           :return cache: Cache, stored for backward computation, of the same shape as x.
           """
           out = None
           cache = None
   		#out用于计算前向传播，cache用于计算反向传播
           out = np.maximum(0,x)
           cache = out
   
           return out, cache
   
       def backward(self, dout, cache):
           """
           :param dout: Upstream gradient from the computational graph, from the Loss function
                       and up to this layer. Has the shape of the output of forward().
           :param cache: The values that were stored during forward() to the memory,
                       to be used during the backpropogation.
           :return: dx: the gradient w.r.t. input X, of the same shape as X
           """
           dx = None
   
           dx = dout
           # dx,cache的维度一样，所以可以直接下面这样获得cache=0的索引
           dx[cache<=0]=0
   
           return dx
   ```

3. **Tanh**

   ```python
   class Tanh:
       def __init__(self):
           pass
   
       def forward(self, x):
           """
           :param x: Inputs, of any shape
   
           :return out: Output, of the same shape as x
           :return cache: Cache, for backward computation, of the same shape as x
           """
           outputs = None
           cache = None
           ########################################################################
           # TODO:                                                                #
           # Implement the forward pass of Tanh activation function               #
   
           outputs = np.tanh(x)
           cache = outputs
    ########################################################################
           #                           END OF YOUR CODE                           #
   
           return outputs, cache
   
       def backward(self, dout, cache):
           """
           :return: dx: the gradient w.r.t. input X, of the same shape as X
           """
           dx = None
           ########################################################################
           # TODO:                                                                #
           # Implement the backward pass of Tanh activation function              #
   
           dx = dout * (1 - np.square(cache))
           ########################################################################
   
           return dx
   ```

4. **LeakyRelu**

   ```python
   class LeakyRelu:
       def __init__(self, slope=0.01):
           self.slope = slope
   
       def forward(self, x):
           """
           :param x: Inputs, of any shape
   
           :return out: Output, of the same shape as x
           :return cache: Cache, for backward computation, of the same shape as x
           """
           outputs = None
           cache = None
           ########################################################################
           # TODO:                                                                #
           # Implement the forward pass of LeakyRelu activation function          #
   
           outputs = np.maximum(self.slope * x, x)
           cache = x
   
           ########################################################################
           #                           END OF YOUR CODE                          #
           return outputs, cache
   
       def backward(self, dout, cache):
           """
           :return: dx: the gradient w.r.t. input X, of the same shape as X
           """
           dx = None
           ########################################################################
           # TODO:                                                                #
           # Implement the backward pass of LeakyRelu activation function         #
   
   
           dx = dout * np.where(cache > 0, 1, self.slope)
   
           ########################################################################
           #                           END OF YOUR CODE                           #
   
           return dx
   ```

   

### 3.2.3 Affine layer仿射层

 Affine layer仿射层又称之为linear 线性变换层， Full-connected Layer全连接层。仿射（Affine）的意思是前面一层中的每一个神经元都连接到当前层中的每一个神经元。仿射层通常被加在卷积神经网络或循环神经网络做出最终预测前的输出的顶层。

**全连接层之前的作用是提取特征，而全连接层的作用是分类**

- 公式:$z=Wx+b$

  - W是权重矩阵

- 反向求导：
  $$
  \begin{aligned}
  &\frac{\partial L}{\partial x}=\frac{\partial L}{\partial z}\cdot W^T\\
  &\frac{\partial L}{\partial W}=x^T\cdot \frac{\partial L}{\partial z}\\
  &\frac{\partial L}{\partial b}=1^T\cdot\frac{\partial L}{\partial z}
  \end{aligned}
  $$

  - $\frac{\partial L}{\partial z}$:对应下面代码的dout
  - $\frac{\partial L}{\partial b}$:是$\frac{\partial L}{\partial z}$即dout的求和
  - $\frac{\partial L}{\partial x},\frac{\partial L}{\partial W}$分别对应下面代码的dx,dw

```python
def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.
    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.
    Inputs:
    :param x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    :param w: A numpy array of weights, of shape (D, M)
    :param b: A numpy array of biases, of shape (M,)
    :return out: output, of shape (N, M)
    :return cache: (x, w, b)
    """
    N, M = x.shape[0], b.shape[0]
    out = np.zeros((N,M))

    #N:每一个minibatch下的样本数
    N=x.shape[0]
    x_shape=x.shape[1:]
    D=1
    for i in range(len(x_shape)):
        D=D*x_shape[i]
    # 将输入x的数据排列为(N,D)的x_new
    x_new = np.reshape(x, (N, D))
    # 前向输出
    out = np.dot(x_new, w) + b

    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.
    Inputs:
    :param dout: Upstream derivative, of shape (N, M)
    :param cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: A numpy array of biases, of shape (M,
    :return dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    :return dw: Gradient with respect to w, of shape (D, M)
    :return db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
	# 将输入x的数据排列为(N,D)的x_new
    N = x.shape[0]
    D_backup = x.shape[1:]
    D = 1
    for i in range(len(D_backup)):
        D = D * D_backup[i]
    x_new = np.reshape(x, (N, D))
	# 反向传播计算
    dw = np.dot(x_new.T, dout)
    db = np.sum(dout, axis=0)
    dx_backup = np.dot(dout, w.T)
    dx = np.reshape(dx_backup, x.shape)

    return dx, dw, db
```

### 3.2.4 N-layer Classification Network

- 用Relu计算一个2层网络,

  ```python
  # Define a dummy input
  test_input = np.random.randn(1, 10)   # (batch_size, input_size)
  
  # Define a test model
  test_model = ClassificationNet(input_size=10, 
                                 hidden_size=128,
                                 activation=Relu(), 
                                 num_layer=2, 
                                 num_classes=3)
  
  # Compute output
  model_output = test_model.forward(test_input)
  ```

  - `ClassificationNet`类的定义如下

- 将上面这些层的前向，反向传播整合在一个类里，就可以想建几层网络建几层，想用什么激活函数用什么激活函数

  ```python
  class ClassificationNet(Network):
      """
      A fully-connected classification neural network with configurable 
      activation function, number of layers, number of classes, hidden size and
      regularization strength. 
      """
  
      def __init__(self, activation=Relu(), num_layer=2,
                   input_size=3 * 32 * 32, hidden_size=100,
                   std=1e-3, num_classes=10, reg=0, **kwargs):
          """
          :param activation: choice of activation function. It should implement
              a forward() and a backward() method.
          :param num_layer: integer, number of layers. 
          :param input_size: integer, the dimension D of the input data.
          :param hidden_size: integer, the number of neurons H in the hidden layer.
          :param std: float, standard deviation used for weight initialization.
          :param num_classes: integer, number of classes.
          :param reg: float, regularization strength.
          """
          super().__init__("cifar10_classification_net")
  
          self.activation = activation
          self.reg_strength = reg
  
          self.cache = None
  
          # Initialize random gaussian weights for all layers and zero bias
          self.num_layer = num_layer
          self.params = {'W1': std * np.random.randn(input_size, hidden_size),
                         'b1': np.zeros(hidden_size)}
  
          for i in range(num_layer - 2):
              self.params['W' + str(i + 2)] = std * np.random.randn(hidden_size,
                                                                    hidden_size)
              self.params['b' + str(i + 2)] = np.zeros(hidden_size)
  
          self.params['W' + str(num_layer)] = std * np.random.randn(hidden_size,
                                                                    num_classes)
          self.params['b' + str(num_layer)] = np.zeros(num_classes)
  
          self.grads = {}
          self.reg = {}
          for i in range(num_layer):
              self.grads['W' + str(i + 1)] = 0.0
              self.grads['b' + str(i + 1)] = 0.0
  
      def forward(self, X):
          """
          Performs the forward pass of the model.
          :param X: Input data of shape (N, d_1, ..., d_k) and contains a minibatch of N
          examples, where each example x[i] has shape (d_1, ..., d_k)
          :return: Predicted value for the data in X, shape N x num_classes
                   num_classes-dimensional array of length N with the classification scores.
          """
  
          self.cache = {}
          self.reg = {}
          X = X.reshape(X.shape[0], -1)
          # Unpack variables from the params dictionary
          for i in range(self.num_layer - 1):
              W, b = self.params['W' + str(i + 1)], self.params['b' + str(i + 1)]
  
              # Forward i_th layer
              X, cache_affine = affine_forward(X, W, b)
              self.cache["affine" + str(i + 1)] = cache_affine
  
              # Activation function
              X, cache_sigmoid = self.activation.forward(X)
              self.cache["sigmoid" + str(i + 1)] = cache_sigmoid
  
              # Store the reg for the current W
              self.reg['W' + str(i + 1)] = np.sum(W ** 2) * self.reg_strength
  
          # last layer contains no activation functions
          W, b = self.params['W' + str(self.num_layer)],\
                 self.params['b' + str(self.num_layer)]
          y, cache_affine = affine_forward(X, W, b)
          self.cache["affine" + str(self.num_layer)] = cache_affine
          self.reg['W' + str(self.num_layer)] = np.sum(W ** 2) * self.reg_strength
  
          return y
  
      def backward(self, dy):
          """
          Performs the backward pass of the model.
          :param dy: N x num_classes array. The gradient wrt the output of the network.
          :return: Gradients of the model output wrt the model weights
          """
  
          # Note that last layer has no activation
          cache_affine = self.cache['affine' + str(self.num_layer)]
          dh, dW, db = affine_backward(dy, cache_affine)
          self.grads['W' + str(self.num_layer)] = \
              dW + 2 * self.reg_strength * self.params['W' + str(self.num_layer)]
          self.grads['b' + str(self.num_layer)] = db
  
          # The rest sandwich layers
          for i in range(self.num_layer - 2, -1, -1):
              # Unpack cache
              cache_sigmoid = self.cache['sigmoid' + str(i + 1)]
              cache_affine = self.cache['affine' + str(i + 1)]
  
              # Activation backward
              dh = self.activation.backward(dh, cache_sigmoid)
  
              # Affine backward
              dh, dW, db = affine_backward(dh, cache_affine)
  
              # Refresh the gradients
              self.grads['W' + str(i + 1)] = dW + 2 * self.reg_strength * \
                                             self.params['W' + str(i + 1)]
              self.grads['b' + str(i + 1)] = db
  
          return self.grads
  
      def save_model(self):
          directory = 'models'
          model = {self.model_name: self}
          if not os.path.exists(directory):
              os.makedirs(directory)
          pickle.dump(model, open(directory + '/' + self.model_name + '.p', 'wb'))
  
      def get_dataset_prediction(self, loader):
          scores = []
          labels = []
  
          for batch in loader:
              X = batch['image']
              y = batch['label']
              score = self.forward(X)
              scores.append(score)
              labels.append(y)
  
          scores = np.concatenate(scores, axis=0)
          labels = np.concatenate(labels, axis=0)
  
          preds = scores.argmax(axis=1)
          acc = (labels == preds).mean()
  
          return labels, preds, acc
  ```

## 3.3加载CIFAR10 Dataset

- 下载数据

    ```python
    # Define output path similar to exercise 3
    i2dl_exercises_path = os.path.dirname(os.path.abspath(os.getcwd()))
    cifar_root = os.path.join(i2dl_exercises_path, "datasets", "cifar10")

    # Dictionary so that we can convert label indices to actual label names
    classes = [
        'plane', 'car', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck',
    ]

    # Simply call dataset class
    dataset = ImageFolderDataset(
            root=cifar_root
        )
    ```

- 这个project中我们的输入是一维的，所以要把数据集展成1维

  ```python
  # Use the Cifar10 mean and standard deviation computed in Exercise 3.
  cifar_mean = np.array([0.49191375, 0.48235852, 0.44673872])
  cifar_std  = np.array([0.24706447, 0.24346213, 0.26147554])
  
  # Define all the transforms we will apply on the images when 
  # retrieving them.
  rescale_transform = RescaleTransform()
  normalize_transform = NormalizeTransform(
      mean=cifar_mean,
      std=cifar_std
  )
  
  # Add the new flatten transform
  flatten_transform = FlattenTransform()
  
  # And string them together
  compose_transform = ComposeTransform([
      rescale_transform, 
      normalize_transform,
      flatten_transform
  ])
  ```
  
- 加载数据

  具体加载代码可看1.4/1.5

  ```python
  from exercise_code.data import MemoryImageFolderDataset
  #DATASET = ImageFolderDataset
  DATASET = MemoryImageFolderDataset
  # Create a dataset and dataloader
  batch_size = 8
  
  dataset = DATASET(
      mode='train',
      root=cifar_root,
      transform=compose_transform,
      split={'train': 0.01, 'val': 0.2, 'test': 0.79}
  )
      
  dataloader = DataLoader(
      dataset=dataset,
      batch_size=batch_size,
      shuffle=True,
      drop_last=True,
  )
  
  print('Dataset size:', len(dataset))
  print('Dataloader size:', len(dataloader))
  ```

## 3.4Cross-Entropy Loss交叉熵

- Cross-Entropy Los模型：

  $$ CE(\hat{y}, y) = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^{C} \Big[ -y_{ik} \log(\hat{y}_{ik}) \Big] $$

  - $ N $ is again the number of samples
  - $ C $ is the number of classes
  - $ \hat{y}_{ik} $ is the probability that the model assigns for the $k$'th class when the $i$'th sample is the input. 
  - $y_{ik} = 1 $ iff the true label of the $i$th sample is $k$ and 0 otherwise. This is called a [one-hot encoding](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/).

- [Logits](https://datascience.stackexchange.com/questions/31041/what-does-logits-in-machine-learning-mean):

  **Logits interpreted to be the unnormalised** (or not-yet normalised) **predictions** (or outputs) **of a model. These can give results, but we don't normally stop with logits, because interpreting their raw values is not easy.**

  比如我们分类2张猫狗图，得到的Logits value分别为：

  1. 第一张图猫为16，狗为0.7，结果判断为猫
  2. 第二张图猫为1.004，狗为0.6，结果判断为猫

  模型都成功了，但我们要想比较这两张图的判断结果，必须要先normalise the logits。为此我们使用softmax function ,它可以将每张图预测结果和为1，并可认为他们是概率。

  第一张图经过normalise（softmax）后，猫的值为0.9999，狗的值为0.0001。所以我们认为这张图99.99%的概率是猫

- Softmax

  - 公式：

    $$softmax(x)=\sigma(x)=\frac{e^{x_i}}{\sum_{i=1}^ne^{x_i}}$$

    - $x=(x_i)_{(1\leq i\leq n)}\in\mathbb{R}^n$是一个vector

  - 问题：数值不稳定Numerical Stability

    计算机中用浮点数来表示实数时总会引入近似误差

    1. **下溢**：当接近零的数被四舍五入为零时发生下溢（除零，对0取对数）
    2. **上溢**：大量级的数被近似为无穷时发生上溢。

  - 解决方法：$$\sigma(x-\max_{1\leq i\leq n}x_i)$$

    - 减去最大值导致exp最大为0，排除了上溢的可能性
    - 分母中至少有一个值为1的项（exp(0)=1），从而也排除了因分母=下溢导致被零除的可能性

- 代码实现：

  ```python
  class CrossEntropyFromLogits(Loss):
      def __init__(self):
          self.cache = {}
       
      def forward(self, y_out, y_truth, reduction='mean'):
          """
          Performs the forward pass of the cross entropy loss function.
          
          :param y_out: [N, C] array with the predicted logits of the model
              (i.e. the value before applying any activation)
          :param y_truth: (N,) array with ground truth labels.
          :return: float, the cross-entropy loss value
          """
          # Transform the ground truth labels into one hot encodings.
          N, C = y_out.shape
          ## Return an array of zeros with the same shape and type as a given array.
          y_truth_one_hot = np.zeros_like(y_out)
          y_truth_one_hot[np.arange(N), y_truth] = 1
          
          # Transform the logits into a distribution using softmax.
          # First make the operation numerically stable by substracting the
          # biggest element from all entries before applying exp
          y_out_exp = np.exp(y_out - np.max(y_out, axis=1, keepdims=True))
          y_out_probs = y_out_exp / np.sum(y_out_exp, axis=1, keepdims=True)
          
          # Compute the loss for each element in the batch.
          # Hint: Why do we add the minus sign to the formula?
          loss = -y_truth_one_hot * np.log(y_out_probs)
          loss = loss.sum(axis=1).mean()
             
          self.cache['probs'] = y_out_probs
          
          return loss
      
      def backward(self, y_out, y_truth):
          """
          Performs the backward pass of the loss function.
  
          :param y_out: [N, C] array predicted value of your model.
                 y_truth: [N, ] array ground truth value of your training set.
          :return: [N, C] array of cross entropy loss gradients w.r.t y_out for
                    each sample of your training set.
          """
          N, C = y_out.shape
          gradient = self.cache['probs']
          gradient[np.arange(N), y_truth] -= 1
          gradient /= N
          
          return gradient
  ```

## 3.5Optimization优化算法

- 注意事项

  Always, always, always when starting a new project or defining a new network: **overfit on a small set first and then generalize**. The 500 images we are using here are already too many sample for most cases. Start with a single sample, then 10 and finally a few hundred. Don't cheap out on this step! More often, your network will fail to generalize properly and you have to first know if it has enough capacity to overfit and that the full training pipeline is working!

- 下面每一点都要先如3.2.4中一样设置我们的模型

### 3.5.1 计算神经网络所需要的内存

经计算计算一个batchsize为8的子集，前向/反向传播分别要用到3M和6M内存。

但我们共有50000个子集，这就需要36.81G的内存。可见SGD重要性

- 计算前向传播需要的内存

  ```python
  # Set up loss
  loss_func = CrossEntropyFromLogits()
  
  # Get a random batch of our dataloader with batch_size 8
  sample_batch = iter(dataloader).__next__()
  sample_images = sample_batch['image']
  sample_labels = sample_batch['label']
  
  # Compute model output
  model_output = model.forward(sample_images)
  
  num_bytes = compute_network_pass_size(model)
  print('\nTotal number of bytes used by network for batch:', GetHumanReadable(num_bytes))
  ```

- 计算反向传播需要的内存

  ```python
  # 1. Compute loss
  _ = loss_func.forward(model_output, sample_labels)
  # 2. Compute loss gradients
  dout = loss_func.backward(model_output, sample_labels)
  # 3. Backpropagate gradients through model
  _ = model.backward(dout)
  
  # Now calculate bytes again
  num_bytes = compute_network_pass_size(model)
  
  print('\nTotal number of bytes used by network for batch:', GetHumanReadable(num_bytes))
  ```

- 用到的`compute_network_pass_size`类

  ```python
  def compute_network_pass_size(model):
      """Computes the size of a network pass in bytes using cached
      parameters as well as gradients"""
      num_bytes = 0
  
      print('Adding layer caches for forward pass:')
      for layer in model.cache.keys():
          # Add size of forward caches
          key_num_bytes = 0
          for value in model.cache[layer]:
              value_num_bytes = sys.getsizeof(value)
              key_num_bytes += value_num_bytes
          num_bytes += key_num_bytes
  
          print(layer, key_num_bytes)
  
      print('\nAdding layer gradients for backward pass:')
      for key in model.grads.keys():
          # Add size of backward gradients
          key_num_bytes = sys.getsizeof(model.grads[key])
          num_bytes += key_num_bytes
          print(key, key_num_bytes)
         
      return num_bytes
  ```

### 3.5.2 SGD

- 使用2.10所写的solver直接调用 SGD优化算法

  ```python
  learning_rate = 1e-2
  
  # We use our training dataloader for validation as well as testing
  solver = Solver(model, dataloader, dataloader, 
                  learning_rate=learning_rate, loss_func=loss_func, optimizer=SGD)
  
  # This might take a while depending on your hardware. When in doubt: use google colab
  solver.train(epochs=20)
  ```

- SGD

  ```python
  class SGD(object):
      def __init__(self, model, loss_func, learning_rate=1e-4):
          self.model = model
          self.loss_func = loss_func
          self.lr = learning_rate
          self.grads = None
  
      def backward(self, y_pred, y_true):
          """
          Compute the gradients wrt the weights of your model
          """
          dout = self.loss_func.backward(y_pred, y_true)
          self.model.backward(dout)
  
      def _update(self, w, dw, lr):
          """
          Update a model parameter
          """
          w -= lr * dw
          return w
  
      def step(self):
          """
          Perform an update step with the update function, using the current
          gradients of the model
          """
  
          # Iterate over all parameters
          for name in self.model.grads.keys():
  
              # Unpack parameter and gradient
              w = self.model.params[name]
              dw = self.model.grads[name]
  
              # Update the parameter
              w_updated = self._update(w, dw, lr=self.lr)
              self.model.params[name] = w_updated
  
              # Reset gradient
              self.model.grads[name] = 0.0
  ```

### 3.5.3 SGD+Momentum

- 可以直接用2.10的olver调用

- SGD+Momentum

  ```python
  class SGDMomentum(object):
      """
      Performs stochastic gradient descent with momentum.
  
      config format:
      - momentum: Scalar between 0 and 1 giving the momentum value.
        Setting momentum = 0 reduces to sgd.
      - velocity: A numpy array of the same shape as w and dw used to store a moving
        average of the gradients.
      """
      def __init__(self, model, loss_func, learning_rate=1e-4, **kwargs):
          self.model = model
          self.loss_func = loss_func
          self.lr = learning_rate
          self.grads = None
          self.optim_config = kwargs.pop('optim_config', {})
          self._reset()
  
      def _reset(self):
          self.optim_configs = {}
          for p in self.model.params:
              d = {k: v for k, v in self.optim_configs.items()}
              self.optim_configs[p] = d
  
      def backward(self, y_pred, y_true):
          """
          Compute the gradients wrt the weights of your model
          """
          dout = self.loss_func.backward(y_pred, y_true)
          self.model.backward(dout)
  
      def _update(self, w, dw, config, lr):
          """
          Update a model parameter
          
          :param w: Current weight matrix
          :param dw: The corresponding calculated gradient, of the same shape as w.
          :param config: A dictionary, containing relevant parameters, such as the "momentum" value. Check it out.
          :param lr: The value of the "learning rate".
  
          :return next_w: The updated value of w.
          :return config: The same dictionary. Might needed to be updated.
          """
          if config is None:
              config = {}
          config.setdefault('momentum', 0.9)
          v = config.get('velocity', np.zeros_like(w))
          next_w = None
  
          ########################################################################
          # TODO: Implement the momentum update formula. Store the updated       #  
          # value in the next_w variable. You should also use and update the     #
          # velocity v.                                                          #
          ########################################################################
  
          mu = config['momentum']
          learning_rate = lr
          v = mu * v - learning_rate * dw
          next_w = w + v
          config['velocity'] = v
  
          ########################################################################
  
          config['velocity'] = v
  
          return next_w, config
  
      def step(self):
          """
          Perform an update step with the update function, using the current
          gradients of the model
          """
  
          # Iterate over all parameters
          for name in self.model.grads.keys():
  
              # Unpack parameter and gradient
              w = self.model.params[name]
              dw = self.model.grads[name]
  
              config = self.optim_configs[name]
  
              # Update the parameter
              w_updated, config = self._update(w, dw, config, lr=self.lr)
              self.model.params[name] = w_updated
              self.optim_configs[name] = config
              # Reset gradient
              self.model.grads[name] = 0.0
  ```

### 3.5.4 Adam

- Adam

  ```python
  class Adam(object):
      """
      Uses the Adam update rule, which incorporates moving averages of both the
      gradient and its square and a bias correction term.
  
      config format:
      - beta1: Decay rate for moving average of first moment of gradient.
      - beta2: Decay rate for moving average of second moment of gradient.
      - epsilon: Small scalar used for smoothing to avoid dividing by zero.
      - m: Moving average of gradient.
      - v: Moving average of squared gradient.
      - t: Iteration number.
      """
      def __init__(self, model, loss_func, learning_rate=1e-4, **kwargs):
          self.model = model
          self.loss_func = loss_func
          self.lr = learning_rate
          self.grads = None
  
          self.optim_config = kwargs.pop('optim_config', {})
  
          self._reset()
  
      def _reset(self):
          self.optim_configs = {}
          for p in self.model.params:
              d = {k: v for k, v in self.optim_configs.items()}
              self.optim_configs[p] = d
  
      def backward(self, y_pred, y_true):
          """
          Compute the gradients wrt the weights of your model
          """
          dout = self.loss_func.backward(y_pred, y_true)
          self.model.backward(dout)
  
      def _update(self, w, dw, config, lr):
          """
          Update a model parameter
          
          :param w: Current weight matrix
          :param dw: The corresponding calculated gradient, of the same shape as w.
          :param config: A dictionary, containing relevant parameters, such as the "beta1" value. Check it out.
          :param lr: The value of the "learning rate".
  
          :return next_w: The updated value of w.
          :return config: The same dictionary. Might needed to be updated.
          """
          if config is None:
              config = {}
          config.setdefault('beta1', 0.9)
          config.setdefault('beta2', 0.999)
          config.setdefault('epsilon', 1e-4)
          config.setdefault('m', np.zeros_like(w))
          config.setdefault('v', np.zeros_like(w))
          config.setdefault('t', 0)
          next_w = None
  
          #########################################################################
          # TODO: Look at the Adam implementation.                                #
          #########################################################################
          m = config['m']
          v = config['v']
          t = config['t']
          beta1 = config['beta1']
          beta2 = config['beta2']
          learning_rate = lr
          eps = config['epsilon']
  
          m = beta1 * m + (1 - beta1) * dw
          m_hat = m / (1 - np.power(beta1, t + 1))
          v = beta2 * v + (1 - beta2) * (dw ** 2)
          v_hat = v / (1 - np.power(beta2, t + 1))
          next_w = w - learning_rate * m_hat / (np.sqrt(v_hat) + eps)
  
          config['t'] = t + 1
          config['m'] = m
          config['v'] = v
          ########################################################################
  
  
          return next_w, config
  
      def step(self):
          """
          Perform an update step with the update function, using the current
          gradients of the model
          """
  
          # Iterate over all parameters
          for name in self.model.grads.keys():
  
              # Unpack parameter and gradient
              w = self.model.params[name]
              dw = self.model.grads[name]
  
              config = self.optim_configs[name]
  
              # Update the parameter
              w_updated, config = self._update(w, dw, config, lr=self.lr)
              self.model.params[name] = w_updated
              self.optim_configs[name] = config
              # Reset gradient
              self.model.grads[name] = 0.0
  ```


# 4. Cifar10分类：调参

## 4.1加载库

```py
# Some lengthy setup.
import matplotlib.pyplot as plt
import numpy as np
import os
import urllib.request

%load_ext autoreload
%autoreload 2
%matplotlib inline

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
```

## 4.2数据预处理

- Rescale调节尺寸，Normalize正则化，Flatten转为1维。来达到我们希望的数据shape

```python
class RescaleTransform:
    """Transform class to rescale images to a given range"""
    def __init__(self, range_=(0, 1), old_range=(0, 255)):
        """
        :param range_: Value range to which images should be rescaled
        :param old_range: Old value range of the images
            e.g. (0, 255) for images with raw pixel values
        """
        self.min = range_[0]
        self.max = range_[1]
        self._data_min = old_range[0]
        self._data_max = old_range[1]

    def __call__(self, images):

        images = images - self._data_min  # normalize to (0, data_max-data_min)
        images /= (self._data_max - self._data_min)  # normalize to (0, 1)
        images *= (self.max - self.min)  # norm to (0, target_max-target_min)
        images += self.min  # normalize to (target_min, target_max)
        
        return images


class NormalizeTransform:
    """
    Transform class to normalize images using mean and std
    Functionality depends on the mean and std provided in __init__():
        - if mean and std are single values, normalize the entire image
        - if mean and std are numpy arrays of size C for C image channels,
            then normalize each image channel separately
    """
    def __init__(self, mean, std):
        """
        :param mean: mean of images to be normalized
            can be a single value, or a numpy array of size C
        :param std: standard deviation of images to be normalized
             can be a single value or a numpy array of size C
        """
        self.mean = mean
        self.std = std

    def __call__(self, images):
        images = (images - self.mean) / self.std
        return images

    
class FlattenTransform:
    """Transform class that reshapes an image into a 1D array"""
    def __call__(self, image):
        return image.flatten()

## 将上面不同模式的数据预处理方式统一
class ComposeTransform:
    """Transform class that combines multiple other transforms into one"""
    def __init__(self, transforms):
        """
        :param transforms: transforms to be combined
        """
        self.transforms = transforms

    def __call__(self, images):
        for transform in self.transforms:
            images = transform(images)
        return images

```

- 加载数据

  ```python
  # Choose your preferred dataset here
  #DATASET = ImageFolderDataset
  DATASET = MemoryImageFolderDataset
  
  # Use the Cifar10 mean and standard deviation computed in Exercise 3.
  cifar_mean = np.array([0.49191375, 0.48235852, 0.44673872])
  cifar_std  = np.array([0.24706447, 0.24346213, 0.26147554])
  
  # Define all the transforms we will apply on the images when 
  # retrieving them.
  rescale_transform = RescaleTransform()
  normalize_transform = NormalizeTransform(
      mean=cifar_mean,
      std=cifar_std
  )
  flatten_transform = FlattenTransform()
  compose_transform = ComposeTransform([rescale_transform, 
                                        normalize_transform,
                                        flatten_transform])
  
  # Create a train, validation and test dataset.
  datasets = {}
  for mode in ['train', 'val', 'test']:
      crt_dataset = DATASET(
          mode=mode,
          root=cifar_root, 
          transform=compose_transform,
          split={'train': 0.6, 'val': 0.2, 'test': 0.2}
      )
      datasets[mode] = crt_dataset
      
  # Create a dataloader for each split.
  dataloaders = {}
  for mode in ['train', 'val', 'test']:
      crt_dataloader = DataLoader(
          dataset=datasets[mode],
          batch_size=256,
          shuffle=True,
          drop_last=True,
      )
      dataloaders[mode] = crt_dataloader
  ```

- 对于训练集trianing set,我们可以**flip the images horizontally** or **blur the images**，来扩充我们的训练集。（不可用于validation and test data）

  ```python
  class RandomHorizontalFlip:
      """
      Transform class that flips an image horizontically randomly with a given probability.
      """
  
      def __init__(self, prob=0.5):
          """
          :param prob: Probability of the image being flipped
          """
          self.p = prob
  
      def __call__(self, image):
          rand = random.uniform(0,1)
          if rand < self.p:
              image = np.flip(image,1)
          return image
      
     
  #Load the data in a dataset without any transformation 
  dataset = DATASET(
          mode=mode,
          root=cifar_root, 
          download_url=download_url,
          split={'train': 0.6, 'val': 0.2, 'test': 0.2},
      )
  
  #Retrieve an image from the dataset and flip it
  image = dataset[1]['image']
  transform = RandomHorizontalFlip(1)
  image_flipped = transform(image)
  ```

- 参见3.2.4的规范方法ClassificationNet，帮助我们方便调参

  ```python
  input_size = datasets['train'][0]['image'].shape[0]
  model = ClassificationNet(input_size=input_size, 
                            hidden_size=512)
                            
  num_layer = 2
  reg = 0.1
  model = ClassificationNet(activation=Sigmoid(), 
                            num_layer=num_layer, 
                            reg=reg,
                            num_classes=10)
  ```

## 4.3 一个分类模型结构

用2.10的`solver`来调用这个model，默认是2层的

```python
class ClassificationNet(Network):
    """
    A fully-connected classification neural network with configurable 
    activation function, number of layers, number of classes, hidden size and
    regularization strength. 
    """

    def __init__(self, activation=Sigmoid(), num_layer=2,
                 input_size=3 * 32 * 32, hidden_size=100,
                 std=1e-3, num_classes=10, reg=0, **kwargs):
        """
        :param activation: choice of activation function. It should implement
            a forward() and a backward() method.
        :param num_layer: integer, number of layers. 
        :param input_size: integer, the dimension D of the input data.
        :param hidden_size: integer, the number of neurons H in the hidden layer.
        :param std: float, standard deviation used for weight initialization.
        :param num_classes: integer, number of classes.
        :param reg: float, regularization strength.
        """
        super(ClassificationNet, self).__init__("cifar10_classification_net")

        self.activation = activation
        self.reg_strength = reg

        self.cache = None

        self.memory = 0
        self.memory_forward = 0
        self.memory_backward = 0
        self.num_operation = 0

        # Initialize random gaussian weights for all layers and zero bias
        self.num_layer = num_layer
        self.params = {'W1': std * np.random.randn(input_size, hidden_size),
                       'b1': np.zeros(hidden_size)}

        for i in range(num_layer - 2):
            self.params['W' + str(i + 2)] = std * np.random.randn(hidden_size,
                                                                  hidden_size)
            self.params['b' + str(i + 2)] = np.zeros(hidden_size)

        self.params['W' + str(num_layer)] = std * np.random.randn(hidden_size,
                                                                  num_classes)
        self.params['b' + str(num_layer)] = np.zeros(num_classes)

        self.grads = {}
        self.reg = {}
        for i in range(num_layer):
            self.grads['W' + str(i + 1)] = 0.0
            self.grads['b' + str(i + 1)] = 0.0

    def forward(self, X):
        """
        Performs the forward pass of the model.

        :param X: Input data of shape N x D. Each X[i] is a training sample.
        :return: Predicted value for the data in X, shape N x 1
                 1-dimensional array of length N with the classification scores.
        """

        self.cache = {}
        self.reg = {}
        X = X.reshape(X.shape[0], -1)
        # Unpack variables from the params dictionary
        for i in range(self.num_layer - 1):
            W, b = self.params['W' + str(i + 1)], self.params['b' + str(i + 1)]

            # Forward i_th layer
            X, cache_affine = affine_forward(X, W, b)
            self.cache["affine" + str(i + 1)] = cache_affine

            # Activation function
            X, cache_sigmoid = self.activation.forward(X)
            self.cache["sigmoid" + str(i + 1)] = cache_sigmoid

            # Store the reg for the current W
            self.reg['W' + str(i + 1)] = np.sum(W ** 2) * self.reg_strength

        # last layer contains no activation functions
        W, b = self.params['W' + str(self.num_layer)],\
               self.params['b' + str(self.num_layer)]
        y, cache_affine = affine_forward(X, W, b)
        self.cache["affine" + str(self.num_layer)] = cache_affine
        self.reg['W' + str(self.num_layer)] = np.sum(W ** 2) * self.reg_strength

        return y

    def backward(self, dy):
        """
        Performs the backward pass of the model.

        :param dy: N x 1 array. The gradient wrt the output of the network.
        :return: Gradients of the model output wrt the model weights
        """

        # Note that last layer has no activation
        cache_affine = self.cache['affine' + str(self.num_layer)]
        dh, dW, db = affine_backward(dy, cache_affine)
        self.grads['W' + str(self.num_layer)] = \
            dW + 2 * self.reg_strength * self.params['W' + str(self.num_layer)]
        self.grads['b' + str(self.num_layer)] = db

        # The rest sandwich layers
        for i in range(self.num_layer - 2, -1, -1):
            # Unpack cache
            cache_sigmoid = self.cache['sigmoid' + str(i + 1)]
            cache_affine = self.cache['affine' + str(i + 1)]

            # Activation backward
            dh = self.activation.backward(dh, cache_sigmoid)

            # Affine backward
            dh, dW, db = affine_backward(dh, cache_affine)

            # Refresh the gradients
            self.grads['W' + str(i + 1)] = dW + 2 * self.reg_strength * \
                                           self.params['W' + str(i + 1)]
            self.grads['b' + str(i + 1)] = db

        return self.grads

    def save_model(self):
        directory = 'models'
        model = {self.model_name: self}
        if not os.path.exists(directory):
            os.makedirs(directory)
        pickle.dump(model, open(directory + '/' + self.model_name + '.p', 'wb'))

    def get_dataset_prediction(self, loader):
        scores = []
        labels = []
        
        for batch in loader:
            X = batch['image']
            y = batch['label']
            score = self.forward(X)
            scores.append(score)
            labels.append(y)
            
        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()

        return labels, preds, acc

```

## 4.4 调参前的准备

- 我们需要先从一个小而简单的结构来调整我们的网络结构。

  As a first step you should try to overfit to a single training sample, then to a few batches of training samples and finally go deeper with larger neural networks and the whole training data.

- 注意用2.10的solver接口来统一调用4.3的model/其他model。

- 通过这一系列操作，可以看看我们的代码是否有问题

  - 1，2：检测能否处理不同数目的数据集
  - 3，4：检测能否处理不同的隐藏层数

1. First, let's start with a 2-layer neural network, and overfit to one single training sample.

   ```python
   from exercise_code.solver import Solver
   from exercise_code.networks.optimizer import SGD, Adam
   from exercise_code.networks import MyOwnNetwork
   ## 调参部分：
   num_layer = 2
   epochs = 20
   reg = 0.1
   batch_size = 4
   
   model = ClassificationNet(num_layer=num_layer, reg=reg)
   # model = MyOwnNetwork()
   
   loss = CrossEntropyFromLogits()
   
   # Make a new data loader with a single training image
   overfit_dataset = DATASET(
       mode='train',
       root=cifar_root, 
       download_url=download_url,
       transform=compose_transform,
       limit_files=1
   )
   dataloaders['train_overfit_single_image'] = DataLoader(
       dataset=overfit_dataset,
       batch_size=batch_size,
       shuffle=True,
       drop_last=False,
   )
   
   # Decrease validation data for only debugging
   debugging_validation_dataset = DATASET(
       mode='val',
       root=cifar_root, 
       download_url=download_url,
       transform=compose_transform,
       limit_files=100
   )
   dataloaders['val_500files'] = DataLoader(
       dataset=debugging_validation_dataset,
       batch_size=batch_size,
       shuffle=True,
       drop_last=True,
   )
   
   solver = Solver(model, dataloaders['train_overfit_single_image'], dataloaders['val_500files'], 
                   learning_rate=1e-3, loss_func=loss, optimizer=Adam)
   solver.train(epochs=epochs)
   
   ## 将train 和validation数据集的loss曲线画出来，可以看出是overfit了吗
   plt.title('Loss curves')
   plt.plot(solver.train_loss_history, '-', label='train')
   plt.plot(solver.val_loss_history, '-', label='val')
   plt.legend(loc='lower right')
   plt.xlabel('Iteration')
   plt.show()
   
   print("Training accuray: %.5f" % (solver.get_dataset_accuracy(dataloaders['train_overfit_single_image'])))
   print("Validation accuray: %.5f" % (solver.get_dataset_accuracy(dataloaders['val_500files'])))
   ```

2. This time we want to overfit to a small set of training batch samples

   ```python
   from exercise_code.networks import MyOwnNetwork
   ## 调参部分：
   num_layer = 2
   epochs = 100
   reg = 0.1
   num_samples = 10
   
   model = ClassificationNet(num_layer=num_layer, reg=reg)
   # model = MyOwnNetwork()
   
   loss = CrossEntropyFromLogits()
   
   # Make a new data loader with a our num_samples training image
   overfit_dataset = DATASET(
       mode='train',
       root=cifar_root, 
       download_url=download_url,
       transform=compose_transform,
       limit_files=num_samples
   )
   dataloaders['train_overfit_10samples'] = DataLoader(
       dataset=overfit_dataset,
       batch_size=batch_size,
       shuffle=True,
       drop_last=False,
   )
   
   solver = Solver(model, dataloaders['train_overfit_10samples'], dataloaders['val_500files'], 
                   learning_rate=1e-3, loss_func=loss, optimizer=Adam)
   solver.train(epochs=epochs)
   
   ## 将train 和validation数据集的loss曲线画出来，可以看出是overfit了吗
   plt.title('Loss curves')
   plt.plot(solver.train_loss_history, '-', label='train')
   plt.plot(solver.val_loss_history, '-', label='val')
   plt.legend(loc='lower right')
   plt.xlabel('Iteration')
   plt.show()
   
   print("Training accuray: %.5f" % (solver.get_dataset_accuracy(dataloaders['train_overfit_10samples'])))
   print("Validation accuray: %.5f" % (solver.get_dataset_accuracy(dataloaders['val_500files'])))
   ```

3. 先看看2层跑的怎么样

   ```python
   from exercise_code.networks import MyOwnNetwork
   
   num_layer = 2
   epochs = 5
   reg = 0.01
   
   # Make a new data loader with 1000 training samples
   num_samples = 1000
   overfit_dataset = DATASET(
       mode='train',
       root=cifar_root, 
       download_url=download_url,
       transform=compose_transform,
       limit_files=num_samples
   )
   dataloaders['train_small'] = DataLoader(
       dataset=overfit_dataset,
       batch_size=batch_size,
       shuffle=True,
       drop_last=False,
   )
   
   
   # Change here if you want to use the full training set
   use_full_training_set = False
   if not use_full_training_set:
       train_loader = dataloaders['train_small']
   else:
       train_loader = dataloaders['train']
       
   
   model = ClassificationNet(num_layer=num_layer, reg=reg)
   # model = MyOwnNetwork()
   
   loss = CrossEntropyFromLogits()
   
   solver = Solver(model, train_loader, dataloaders['val'], 
                   learning_rate=1e-3, loss_func=loss, optimizer=Adam)
   
   solver.train(epochs=epochs)
   
   ## 效仿上面1，2画loss曲线
   ```

4. 再看看5层跑的怎么样

   ```python
   from exercise_code.networks import MyOwnNetwork
   
   num_layer = 5
   epochs = 5
   reg = 0.01
   
   model = ClassificationNet(num_layer=num_layer, reg=reg)
   # model = MyOwnNetwork()
   
   # Change here if you want to use the full training set
   use_full_training_set = False
   if not use_full_training_set:
       train_loader = dataloaders['train_small']
   else:
       train_loader = dataloaders['train']
   
   loss = CrossEntropyFromLogits()
   
   solver = Solver(model, train_loader, dataloaders['val'], 
                   learning_rate=1e-3, loss_func=loss, optimizer=Adam)
   
   solver.train(epochs=epochs)
   ```

## 4.5调参方法

### 4.5.1Grid Search

- 代码： 

  ```python
  def grid_search(train_loader, val_loader,
                  grid_search_spaces = {
                      "learning_rate": [0.0001, 0.001, 0.01, 0.1],
                      "reg": [1e-4, 1e-5, 1e-6]
                  },
                  model_class=ClassificationNet, epochs=20, patience=5):
      """
      A simple grid search based on nested loops to tune learning rate and
      regularization strengths.
      Keep in mind that you should not use grid search for higher-dimensional
      parameter tuning, as the search space explodes quickly.
  
      Required arguments:
          - train_dataloader: A generator object returning training data
          - val_dataloader: A generator object returning validation data
  
      Optional arguments:
          - grid_search_spaces: a dictionary where every key corresponds to a
          to-tune-hyperparameter and every value contains a list of possible
          values. Our function will test all value combinations which can take
          quite a long time. If we don't specify a value here, we will use the
          default values of both our chosen model as well as our solver
          - model: our selected model for this exercise
          - epochs: number of epochs we are training each model
          - patience: if we should stop early in our solver
  
      Returns:
          - The best performing model
          - A list of all configurations and results
      """
      configs = []
  
      """
      # Simple implementation with nested loops
      for lr in grid_search_spaces["learning_rate"]:
          for reg in grid_search_spaces["reg"]:
              configs.append({"learning_rate": lr, "reg": reg})
      """
  
      # More general implementation using itertools
      for instance in product(*grid_search_spaces.values()):
          configs.append(dict(zip(grid_search_spaces.keys(), instance)))
  
      return findBestConfig(train_loader, val_loader, configs, epochs, patience,
                            model_class)
  ```

- 调用：

  ```python
  from exercise_code.networks import MyOwnNetwork
  
  # Specify the used network
  model_class = ClassificationNet
  
  from exercise_code import hyperparameter_tuning
  best_model, results = hyperparameter_tuning.grid_search(
      dataloaders['train_small'], dataloaders['val_500files'],
      grid_search_spaces = {
          "learning_rate": [1e-2, 1e-3, 1e-4], 
          "reg": [1e-4]
      },
      model_class=model_class,
      epochs=10, patience=5)
  ```

### 4.5.2 Random Search

- 代码

  ```python
  def random_search(train_loader, val_loader,
                    random_search_spaces = {
                        "learning_rate": ([0.0001, 0.1], 'log'),
                        "hidden_size": ([100, 400], "int"),
                        "activation": ([Sigmoid(), Relu()], "item"),
                    },
                    model_class=ClassificationNet, num_search=20, epochs=20,
                    patience=5):
      """
      Samples N_SEARCH hyper parameter sets within the provided search spaces
      and returns the best model.
  
      See the grid search documentation above.
  
      Additional/different optional arguments:
          - random_search_spaces: similar to grid search but values are of the
          form
          (<list of values>, <mode as specified in ALLOWED_RANDOM_SEARCH_PARAMS>)
          - num_search: number of times we sample in each int/float/log list
      """
      configs = []
      for _ in range(num_search):
          configs.append(random_search_spaces_to_config(random_search_spaces))
  
      return findBestConfig(train_loader, val_loader, configs, epochs, patience,
                            model_class)
  
      
  def random_search_spaces_to_config(random_search_spaces):
      """"
      Takes search spaces for random search as input; samples accordingly
      from these spaces and returns the sampled hyper-params as a config-object,
      which will be used to construct solver & network
      """
      
      config = {}
  
      for key, (rng, mode)  in random_search_spaces.items():
          if mode not in ALLOWED_RANDOM_SEARCH_PARAMS:
              print("'{}' is not a valid random sampling mode. "
                    "Ignoring hyper-param '{}'".format(mode, key))
          elif mode == "log":
              if rng[0] <= 0 or rng[-1] <=0:
                  print("Invalid value encountered for logarithmic sampling "
                        "of '{}'. Ignoring this hyper param.".format(key))
                  continue
              sample = random.uniform(log10(rng[0]), log10(rng[-1]))
              config[key] = 10**(sample)
          elif mode == "int":
              config[key] = random.randint(rng[0], rng[-1])
          elif mode == "float":
              config[key] = random.uniform(rng[0], rng[-1])
          elif mode == "item":
              config[key] = random.choice(rng)
  
      return config
  ```

- 调用：

  ```python
  from exercise_code.hyperparameter_tuning import random_search
  from exercise_code.networks import MyOwnNetwork
  
  # Specify the used network
  model_class = ClassificationNet
  
  best_model, results = random_search(
      dataloaders['train_small'], dataloaders['val_500files'],
      random_search_spaces = {
          "learning_rate": ([1e-2, 1e-6], 'log'),
          "reg": ([1e-3, 1e-7], "log"),
          "loss_func": ([CrossEntropyFromLogits()], "item")
      },
      model_class=model_class,
      num_search = 1, epochs=20, patience=5)
  ```

### 4.5.3 找到最好的那一组数据：

- 通过计算train loss, val loss来得到：

  ```python
  def findBestConfig(train_loader, val_loader, configs, EPOCHS, PATIENCE,
                     model_class):
      """
      Get a list of hyperparameter configs for random search or grid search,
      trains a model on all configs and returns the one performing best
      on validation set
      """
      
      best_val = None
      best_config = None
      best_model = None
      results = []
      
      for i in range(len(configs)):
          print("\nEvaluating Config #{} [of {}]:\n".format(
              (i+1), len(configs)),configs[i])
  
          model = model_class(**configs[i])
          solver = Solver(model, train_loader, val_loader, **configs[i])
          solver.train(epochs=EPOCHS, patience=PATIENCE)
          results.append(solver.best_model_stats)
  
          if not best_val or solver.best_model_stats["val_loss"] < best_val:
              best_val, best_model,\
              best_config = solver.best_model_stats["val_loss"], model, configs[i]
              
      print("\nSearch done. Best Val Loss = {}".format(best_val))
      print("Best Config:", best_config)
      return best_model, list(zip(configs, results))
  ```

### 4.5.4 Early stopping

Usually, at some point the validation loss goes up again, which is a sign that we're overfitting to our training data. Since it actually doesn't make sense to train further at this point, it's common practice to apply "early stopping", i.e., cancel the training process when the validation loss doesn't improve anymore. The nice thing about this concept is, that not only it improves generalization through the prevention of overfitting, but also it saves us a lot of time - one of our most valuable resources in deep learning.

Since there are natural fluctuations in the validation loss, you usually don't cancel the training process right at the first epoch when the validation-loss increases, but instead, you wait for some epochs **(specified by the `patience`-parameter)** and if the loss still doesn't improve, we stop.

**实现代码见2.10**



## 4.6 实际调参：

策略：At the beginning, it's a good approach to first do a coarse random search across a **wide range of values** to find promising sub-ranges of your parameter space and use **a medium large subset of the dataset** . Afterwards, you can zoom into these ranges and do another random search (or grid search) to finetune the configurations. Use the cell below to play around and find good hyperparameters for your model!

```python
from exercise_code.networks import MyOwnNetwork

best_model = ClassificationNet()
#best_model = MyOwnNetwork()

########################################################################
# TODO:                                                                #
# Implement your own neural network and find suitable hyperparameters  #
# Be sure to edit the MyOwnNetwork class in the following code snippet #
# to upload the correct model!                                         #
########################################################################
model_class = ClassificationNet

best_model, results = random_search(
    dataloaders['train_small'], dataloaders['val_500files'],
    random_search_spaces = {
        "learning_rate": ([1e-2, 1e-6], 'log'),
        "reg": ([1e-3, 1e-7], "log"),
        "loss_func": ([CrossEntropyFromLogits()], "item"),
        "num_layer":([3,4],"int")
    },
    model_class=model_class,
    num_search = 4, epochs=20, patience=5)


########################################################################
#                           END OF YOUR CODE                           #
########################################################################
```

# 5. 用PyTorch Lighhtning来Cifar10分类

使用PyTorch Lightning可以方便的调用各种硬件资源以及调试程序。让我们专注于模型的改进。

## 5.1 加载库

```python
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
%load_ext autoreload
%autoreload 2

# 使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
```

## 5.2 使用TensorBoard调试

Anaconda的shell中输

```
tensorboard --logdir lightning_logs --port 6006
```

## 5.3 用PL定义模型

### 5.3.1 数据加载

PyTorch Lightning 提供 `LightningDataModule` 作为基类，来设置dataloaders数据加载.

```python
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.opt = hparams
        if 'loading_method' not in hparams.keys():
            self.opt['loading_method'] = 'Image'
        if 'num_workers' not in hparams.keys():
            self.opt['num_workers'] = 2

    def prepare_data(self, stage=None, CIFAR_ROOT="../datasets/cifar10"):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # create dataset
        CIFAR_ROOT = "../datasets/cifar10"
        my_transform = None
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # 预处理数据（PyTorch的函数方法)
        # transforms.Compose()建立一些预处理方法来处理数据
        # transforms.ToTensor()讲图片或数组从(0,255)转为(0,1)的Tensor（归一化）
        # transforms.Normalize(mean,std)用提供的平均数，标准差来去均值实现中心化的处理（标准化）
        my_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ])
        
        # Make sure to use a consistent transform for validation/test
        train_val_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

        # Note: you can change the splits if you want :)
        split = {
            'train': 0.6,
            'val': 0.2,
            'test': 0.2
        }
        split_values = [v for k,v in split.items()]
        assert sum(split_values) == 1.0
        
        ## 划分数据并进行数据预处理
        if self.opt['loading_method'] == 'Image':
            # Set up a full dataset with the two respective transforms
            cifar_complete_augmented = torchvision.datasets.ImageFolder(root=CIFAR_ROOT, transform=my_transform)
            cifar_complete_train_val = torchvision.datasets.ImageFolder(root=CIFAR_ROOT, transform=train_val_transform)

            # Instead of splitting the dataset in the beginning you can also # split using a sampler. This is not better, but we wanted to show it off here as an example by using the default ImageFolder dataset :)

            # First regular splitting which we did for you before
            N = len(cifar_complete_augmented)        
            num_train, num_val = int(N*split['train']), int(N*split['val'])
            indices = np.random.permutation(N)
            train_idx, val_idx, test_idx = indices[:num_train], indices[num_train:num_train+num_val], indices[num_train+num_val:]

            # Now we can set the sampler via the respective subsets
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            test_sampler= SubsetRandomSampler(test_idx)
            self.sampler = {"train": train_sampler, "val": val_sampler, "test": test_sampler}

            # assign to use in dataloaders
            self.dataset = {}
            self.dataset["train"], self.dataset["val"], self.dataset["test"] = cifar_complete_augmented,\
                cifar_complete_train_val, cifar_complete_train_val

        elif self.opt['loading_method'] == 'Memory':
            self.dataset = {}
            self.sampler = {}

            for mode in ['train', 'val', 'test']:
                # Set transforms
                if mode == 'train':
                    transform = my_transform
                else:
                    transform = train_val_transform

                self.dataset[mode] = MemoryImageFolderDataset(
                    root = CIFAR_ROOT,
                    transform = transform,
                    mode = mode,
                    split = split
                )
        else:
            raise NotImplementedError("Wrong loading method")

    def return_dataloader_dict(self, mode):
        arg_dict = {
            'batch_size': self.opt["batch_size"],
            'num_workers': self.opt['num_workers'],
            'persistent_workers': True,
            'pin_memory': True
        }
        if self.opt['loading_method'] == 'Image':
            arg_dict['sampler'] = self.sampler[mode]
        elif self.opt['loading_method'] == 'Memory':
            arg_dict['shuffle'] = True if mode == 'train' else False
        return arg_dict

    def train_dataloader(self):
        arg_dict = self.return_dataloader_dict('train')
        return DataLoader(self.dataset["train"], **arg_dict)

    def val_dataloader(self):
        arg_dict = self.return_dataloader_dict('val')
        return DataLoader(self.dataset["val"], **arg_dict)
    
    def test_dataloader(self):
        arg_dict = self.return_dataloader_dict('train')
        return DataLoader(self.dataset["train"], **arg_dict)
```

- Dataloader

   `DataLoader` class that is used to create `train_dataloader` and `val_dataloader`

  ```python
  import numpy as np
  class DataLoader:
      """
      Dataloader Class
      Defines an iterable batch-sampler over a given dataset
      """
      def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
          """
          :param dataset: dataset from which to load the data
          :param batch_size: how many samples per batch to load
          :param shuffle: set to True to have the data reshuffled at every epoch
          :param drop_last: set to True to drop the last incomplete batch,
              if the dataset size is not divisible by the batch size.
              If False and the size of dataset is not divisible by the batch
              size, then the last batch will be smaller.
          """
          self.dataset = dataset
          self.batch_size = batch_size
          self.shuffle = shuffle
          self.drop_last = drop_last
  
      def __iter__(self):
          def combine_batch_dicts(batch):
              """
              Combines a given batch (list of dicts) to a dict of numpy arrays
              :param batch: batch, list of dicts
                  e.g. [{k1: v1, k2: v2, ...}, {k1:, v3, k2: v4, ...}, ...]
              :returns: dict of numpy arrays
                  e.g. {k1: [v1, v3, ...], k2: [v2, v4, ...], ...}
              """
              batch_dict = {}
              for data_dict in batch:
                  for key, value in data_dict.items():
                      if key not in batch_dict:
                          batch_dict[key] = []
                      batch_dict[key].append(value)
              return batch_dict
  
          def batch_to_numpy(batch):
              """Transform all values of the given batch dict to numpy arrays"""
              numpy_batch = {}
              for key, value in batch.items():
                  numpy_batch[key] = np.array(value)
              return numpy_batch
  
          if self.shuffle:
              index_iterator = iter(np.random.permutation(len(self.dataset)))
          else:
              index_iterator = iter(range(len(self.dataset)))
  
          batch = []
          for index in index_iterator:
              batch.append(self.dataset[index])
              if len(batch) == self.batch_size:
                  yield batch_to_numpy(combine_batch_dicts(batch))
                  batch = []
  
          if len(batch) > 0 and not self.drop_last:
              yield batch_to_numpy(combine_batch_dicts(batch))
  
      def __len__(self):
          length = None
  
          if self.drop_last:
              length = len(self.dataset) // self.batch_size
          else:
              length = int(np.ceil(len(self.dataset) / self.batch_size))
  
          return length
  ```

### 5.3.2 神经网络结构

PyTorch Lightning的神经网络模型是基于`pl.LightningModule`类来创建的

根据[torch.nn](https://pytorch.org/docs/stable/nn.html)来定义我们的模型。

根据[torch.optim](https://pytorch.org/docs/stable/optim.html)来定义我们的优化器

```python
class MyPytorchModel(pl.LightningModule):
    
    def __init__(self, hparams):
        super().__init__()

        # set hyperparams
        self.save_hyperparameters(hparams)
        ## 定义神经网络模型
        self.model = None
        self.model = nn.Sequential(
            nn.Linear(self.hparams["input_size"],self.hparams["nn_hidden_Layer1"]),
            nn.ReLU(),
            nn.Linear(self.hparams["nn_hidden_Layer1"],self.hparams["num_classes"]),
            nn.ReLU()
        )

    def forward(self, x):

        # x.shape = [batch_size, 3, 32, 32] -> flatten the image first
        x = x.view(x.shape[0], -1)

        # feed x into model!
        x = self.model(x)

        return x
    
    def general_step(self, batch, batch_idx, mode):
        images, targets = batch

        # forward pass
        out = self.forward(images)

        # loss
        loss = F.cross_entropy(out, targets)

        preds = out.argmax(axis=1)
        n_correct = (targets == preds).sum()
        n_total = len(targets)
        return loss, n_correct, n_total
    
    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        length = sum([x[mode + '_n_total'] for x in outputs])
        total_correct = torch.stack([x[mode + '_n_correct'] for x in outputs]).sum().cpu().numpy()
        acc = total_correct / length
        return avg_loss, acc

    def training_step(self, batch, batch_idx):
        loss, n_correct, n_total = self.general_step(batch, batch_idx, "train")
        self.log('loss',loss)
        return {'loss': loss, 'train_n_correct':n_correct, 'train_n_total': n_total}

    def validation_step(self, batch, batch_idx):
        loss, n_correct, n_total = self.general_step(batch, batch_idx, "val")
        self.log('val_loss',loss)
        return {'val_loss': loss, 'val_n_correct':n_correct, 'val_n_total': n_total}
    
    def test_step(self, batch, batch_idx):
        loss, n_correct, n_total = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'test_n_correct':n_correct, 'test_n_total': n_total}

    def validation_epoch_end(self, outputs):
        avg_loss, acc = self.general_end(outputs, "val")
        self.log('val_loss',avg_loss)
        self.log('val_acc',acc)
        return {'val_loss': avg_loss, 'val_acc': acc}

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(),self.hparams["learning_rate"], weight_decay=self.hparams['weight_decay'])
        StepLR = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[30], gamma=0.5)
        return optim

    def getTestAcc(self, loader):
        self.model.eval()
        self.model = self.model.to(self.device)

        scores = []
        labels = []

        for batch in tqdm(loader):
            X, y = batch
            X = X.to(self.device)
            score = self.forward(X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc

```



## 5.4 调参

```python
from exercise_code.MyPytorchModel import MyPytorchModel, CIFAR10DataModule
# make sure you have downloaded the Cifar10 dataset on root: "../datasets/cifar10", if not, please check exercise 03.
hparams = {
    "loading_method":"Memory",
    "num_workers":1,
    "input_size":3*3*32,
    "batch_size":1000,
    "learning_rate":5e-5,
    "weight_decay":1e-3,
    "nn_hidden_Layer1":1500,
    "num_classes":10
} 

# Make sure you downloaded the CIFAR10 dataset already when using this cell
# since we are showcasing the pytorch inhering ImageFolderDataset that
# doesn't automatically download our data. Check exercise 3

# If you want to switch to the memory dataset instead of image folder use
# hparams["loading_method"] = 'Memory'
# The default is hparams["loading_method"] = 'Image'
# You will notice that it takes way longer to initialize a MemoryDataset
# method because we have to load the data points into memory all the time.

# You might get warnings below if you use too few workers. Pytorch uses
# a more sophisticated Dataloader than the one you implemented previously.
# In particular it uses multi processing to have multiple cores work on
# individual data samples. You can enable more than workers (default=2)
# via 
# hparams['num_workers'] = 8

# Set up the data module including your implemented transforms
data_module = CIFAR10DataModule(hparams)
data_module.prepare_data()
# Initialize our model
model = MyPytorchModel(hparams)
```

## 5.5训练

使用pytorch_lightning的[trainer](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html)

```python
import pytorch_lightning as pl
trainer = None

trainer = pl.Trainer(
    max_epochs=2,
    accelerator = 'gpu' 
)

trainer.fit(model, data_module)
```

# 6.Autoencoder for MNIST in PyTorch Lightning

## 6.1 加载库

```python
# 魔法函数
%load_ext autoreload
%autoreload 2
%matplotlib inline
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import pytorch_lightning as pl
from exercise_code.image_folder_dataset import ImageFolderDataset
from pytorch_lightning.loggers import TensorBoardLogger
torch.manual_seed(42)
# 使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# 使用TensorBoard调参
%load_ext tensorboard
%tensorboard --logdir lightning_logs --port 6006
```

## 6.2下载数据集

```python
# 图像预处理(PyTorch)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

i2dl_exercises_path = os.path.dirname(os.path.abspath(os.getcwd()))
mnist_root = os.path.join(i2dl_exercises_path, "datasets", "mnist")

# 下载带标签的数据集
train = ImageFolderDataset(root=mnist_root,images='train_images.pt',labels='train_labels.pt',force_download=False,verbose=True,transform=transform)
val = ImageFolderDataset(root=mnist_root,images='val_images.pt',labels='val_labels.pt',force_download=False,verbose=True,transform=transform)
test = ImageFolderDataset(root=mnist_root,images='test_images.pt',labels='test_labels.pt',force_download=False,verbose=True,transform=transform)

# 下载不带标签的数据集
# We also set up the unlabeled images which we will use later
unlabeled_train = ImageFolderDataset(root=mnist_root,images='unlabeled_train_images.pt',force_download=False,verbose=True,transform=transform)
unlabeled_val = ImageFolderDataset(root=mnist_root,images='unlabeled_val_images.pt',force_download=False,verbose=True,transform=transform)
```

- 看几个图片，看图片和label是否配对

  ```python
  plt.rcParams['figure.figsize'] = (6,6) # Make the figures a bit bigger
  
  for i in range(9):
      image = np.array(train[i][0].squeeze()) # get the image of the data sample
      label = train[i][1] # get the label of the data sample
      plt.subplot(3,3,i+1)
      plt.imshow(image, cmap='gray', interpolation='none')
      plt.title("Class {}".format(label))
      
  plt.tight_layout()
  print('The shape of our greyscale images: ', image.shape)
  ```


## 6.3 一个简单的分类器

用于分类图片是0-9中的某一个数字

```python
class Classifier(pl.LightningModule):

    def __init__(self, hparams, encoder, train_set=None, val_set=None, test_set=None):
        super().__init__()
        # set hyperparams
        self.save_hyperparameters(hparams, ignore=['encoder'])
        self.encoder = encoder
        self.model = nn.Identity()
        self.data = {'train': train_set,
                     'val': val_set,
                     'test': test_set}
        self.model = nn.Sequential(
            nn.Linear(20,self.hparams["hidden_size_2"]),
            nn.LeakyReLU(),
            nn.Linear(self.hparams["hidden_size_2"],self.hparams["output_size"]),
        )

    def forward(self, x):
        # 这里的变量都是pytorch的tensor类型
        # 先调用encoder类来将图片有意义的地方提取出来
        x = self.encoder(x)
        # 再调用classifier来区分数字
        x = self.model(x)
        return x

    def general_step(self, batch, batch_idx, mode):
        # 获得样本和标签，tensor类型
        images, targets = batch
        # tensor.shape()返回tensor的形状
        # tensor.view()把数据变形成nxm的数据，-1表示这个位置由其他位置的的数字推断而出。
        # 例：一个tensor是2X3的，排成一维就是tensor.view(1,6),排成3x2就是tensor.view(3,2)
        # 这里images.shape[0]就是n个样本，第二个-1自动推断出D个特征
        flattened_images = images.view(images.shape[0], -1)
        # forward pass
        out = self.forward(flattened_images)
        # loss
        loss = F.cross_entropy(out, targets)
        preds = out.argmax(axis=1)
        n_correct = (targets == preds).sum()
        return loss, n_correct

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        total_correct = torch.stack(
            [x[mode + '_n_correct'] for x in outputs]).sum().cpu().numpy()
        acc = total_correct / len(self.data[mode])
        return avg_loss, acc

    def training_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "train")
        self.log("train_loss_cls", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "val")
        return {'val_loss': loss, 'val_n_correct': n_correct}

    def test_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'test_n_correct': n_correct}

    def validation_end(self, outputs):
        avg_loss, acc = self.general_end(outputs, "val")
        self.log("val_loss", avg_loss)
        self.log("val_acc", acc)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.data['train'], shuffle=True, batch_size=self.hparams['batch_size'])

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.data['val'], batch_size=self.hparams['batch_size'])

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.data['test'], batch_size=self.hparams['batch_size'])

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(),self.hparams["learning_rate"],weight_decay = self.hparams['weight_decay'])
        return optim

    def getAcc(self, loader=None):
        self.eval()
        self = self.to(self.device)

        if not loader:
            loader = self.test_dataloader()

        scores = []
        labels = []

        for batch in loader:
            X, y = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            score = self.forward(flattened_X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc
```

## 6.4 Autoencoder

图片需要认为label后才能用于训练，这会花费大量的人力。一种解决方式是Data Augmentation

另一种方法是：**transfer learning**

- 首先我们用没有标记过的图片来训练Encoder-Decoder。这样encoder的神经网络就具备了用低维度latent space来表示原图像的能力

  1. Encoder:
     - The `encoder`'s task is to extract meaningful information out of our input so that the classifier can make a proper decision.

  2. Decoder:
     - 将encoder中压缩的图片，再还原成原图
  3. 因为是没有标记过的图片所以我们的loss function需要改成mean squared error between our input pixels and output pixels.
     - 其实就是对比原图和还原图

- 接着我们再用标记过的图片来训练Encoder-Decoder。这样encoder神经网络的权重我们就可以直接使用上面的权重，只需要训练classifier的权重了

### 6.4.1 Encoder

```python
class Encoder(nn.Module):

    def __init__(self, hparams, input_size=28 * 28, latent_dim=20):
        super().__init__()

        # set hyperparams
        self.latent_dim = latent_dim
        self.input_size = input_size
        self.hparams = hparams
        self.encoder = None

        self.encoder = nn.Sequential(
            nn.Linear(self.input_size,self.hparams["hidden_size"]),
            nn.LeakyReLU(),
            nn.Linear(self.hparams["hidden_size"],latent_dim)
        )

    def forward(self, x):
        # feed x into encoder!
        return self.encoder(x)
```

### 6.4.2 Decoder

```python
class Decoder(nn.Module):

    def __init__(self, hparams, latent_dim=20, output_size=28 * 28):
        super().__init__()

        # set hyperparams
        self.hparams = hparams
        self.decoder = None

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim,self.hparams["hidden_size"]),
            nn.LeakyReLU(),
            nn.Linear(self.hparams["hidden_size"],output_size)
        )

    def forward(self, x):
        # feed x into decoder!
        return self.decoder(x)
```

### 6.4.3 AutoEncoder

```python
class Autoencoder(pl.LightningModule):

    def __init__(self, hparams, encoder, decoder, train_set, val_set):
        super().__init__()
        # set hyperparams
        self.save_hyperparameters(hparams, ignore=['encoder', 'decoder'])

        # Define models
        self.encoder = encoder
        self.decoder = decoder
        self.train_set = train_set
        self.val_set = val_set

    def forward(self, x):
        reconstruction = None
        x = self.encoder(x)
        reconstruction = self.decoder(x)
        return reconstruction

    def general_step(self, batch, batch_idx, mode):
        images = batch
        flattened_images = images.view(images.shape[0], -1)

        # forward pass
        reconstruction = self.forward(flattened_images)

        # loss
        loss = F.mse_loss(reconstruction, flattened_images)

        return loss, reconstruction

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        return avg_loss

    def training_step(self, batch, batch_idx):
        loss, _ = self.general_step(batch, batch_idx, "train")
        self.log("train_loss_ae", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch
        flattened_images = images.view(images.shape[0], -1)

        reconstruction = self.forward(flattened_images)
        loss = F.mse_loss(reconstruction, flattened_images)

        reconstruction = reconstruction.view(
            reconstruction.shape[0], 28, 28).cpu().numpy()
        images = np.zeros((len(reconstruction), 3, 28, 28))
        for i in range(len(reconstruction)):
            images[i, 0] = reconstruction[i]
            images[i, 2] = reconstruction[i]
            images[i, 1] = reconstruction[i]
        self.logger.experiment.add_images(
            'reconstructions', images, self.current_epoch, dataformats='NCHW')
        return loss

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, shuffle=True, batch_size=self.hparams['batch_size'])

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.hparams['batch_size'])

    def configure_optimizers(self):

        optim = None
        self.model = nn.Sequential(self.encoder,self.decoder)
        # optim = torch.optim.SGD(self.model.parameters(), self.hparams["learning_rate"], momentum=0.9)
        optim = torch.optim.Adam(self.model.parameters(),self.hparams["learning_rate"],weight_decay = self.hparams['weight_decay'])
        return optim

    def getReconstructions(self, loader=None):
        self.eval()
        self = self.to(self.device)

        if not loader:
            loader = self.val_dataloader()

        reconstructions = []

        for batch in loader:
            X = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            reconstruction = self.forward(flattened_X)
            reconstructions.append(
                reconstruction.view(-1, 28, 28).cpu().detach().numpy())

        return np.concatenate(reconstructions, axis=0)
```



