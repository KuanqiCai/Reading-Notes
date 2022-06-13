# *. 各种库的用法

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

## 1）损失函数Loss Functions

具体代码参见2.4

- Loss Function: 

  Used to measure the goodness of the predictions(the network's performance)

- Classification loss:

  - Cross-Entropy loss 交叉熵损失函数

    也叫做Softmax Loss，用于多分类中表示预测和真实分布之间的距离有多远。

    $$ CE(\hat{y}, y) = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^{C} \Big[ -y_{ik} \log(\hat{y}_{ik}) \Big] $$

    where:
    - $ N $ is again the number of samples

    - $ C $ is the number of classes
      - 当C=2，即binary cross-entropy(BCE)
    - $ \hat{y}_{ik} $ is the probability that the model assigns for the $k$'th class when the $i$'th sample is the input. 
    - $y_{ik} = 1 $ iff the true label of the $i$th sample is $k$ and 0 otherwise. This is called a [one-hot encoding](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/).

  - Loss function: Binary Cross-Entropy (BCE).用于二分类

    $$BCE(y,\hat{y}) =- y \cdot log(\hat y ) - (1- y) \cdot log(1-\hat y) $$

    - $y\in\mathbb{R}$ is the ground truth
    -  $\hat y\in\mathbb{R}$ is the predicted probability of the house being expensive.

- Regression loss

  - L1 loss: $L(y,\hat{y};\theta)=\frac{1}{n}\sum_i^n||y_i-\hat y_i||_1$
  - MSE loss: $L(y,\hat{y};\theta)=\frac{1}{n}\sum_i^n||y_i-\hat{y}_i||_2^2$

## 2) 激活函数Activation Functions

- Sigmoid: $\sigma(x)=\frac{1}{(1+e^{-x})}$

  Sigmoid is one of the oldest used non-linearities. 

  - 公式：$\sigma(x)=\frac{1}{1+exp(-x)}$

  - Sigmoid的导数：$\frac{\partial\sigma(x)}{\partial x}=\sigma(x)(1-\sigma(x))$

  - 反向求导：
    $$
    \frac{\partial L}{\partial x}=\frac{\partial L}{\partial \sigma(x)}\cdot\frac{\partial\sigma(x)}{\partial x}
    $$

    - $\frac{\partial L}{\partial \sigma(x)}$对应下面3.2.2代码backward中的`dout`
    - $\frac{\partial\sigma(x)}{\partial x}$对应下面3.2.2代码backward中的`sd`

- tanh: $tanh(x)$

- ReLU:$max(0,x)$

  Rectified Linear Units线性整流单元 are the currently most used non-linearities in deep learning

  - 公式:$ReLU(x) = max(0,x)=\begin{cases}x(x>0)\\0(x\leq0) \end{cases}$

    - ReLU的导数：$\frac{\partial ReLU(x)}{\partial x}=\begin{cases}1(x>0)\\0(x\leq0) \end{cases}$

  - 反向求导:
    $$
    \frac{\partial L}{\partial x}=\frac{\partial L}{\partial ReLU(x)}\cdot\frac{\partial ReLU(x)}{\partial x}
    $$

    - $\frac{\partial L}{\partial ReLU(x)}$对应下面3.2.2代码backward中的dout
    - $\frac{\partial ReLU(x)}{\partial x}$如果>0值为1就是传dout本身，所以只需要让<=0的dout值为0即可。

- Leaky ReLU: $max(0.1x,x)$

## 3) 优化算法Optimization 

具体代码实现见3.5

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

# 2.逻辑回归：判断房家贵/不贵

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


from exercise_code.networks.optimizer import Optimizer


class Solver(object):
    """
    A Solver encapsulates all the logic necessary for training classification
    or regression models.
    The Solver performs gradient descent using the given learning rate.

    The solver accepts both training and validation data and labels so it can
    periodically check classification accuracy on both training and validation
    data to watch out for overfitting.

    To train a model, you will first construct a Solver instance, passing the
    model, dataset, learning_rate to the constructor.
    You will then call the train() method to run the optimization
    procedure and train the model.

    After the train() method returns, model.W will contain the parameters
    that performed best on the validation set over the course of training.
    In addition, the instance variable solver.loss_history will contain a list
    of all losses encountered during training and the instance variables
    solver.train_loss_history and solver.val_loss_history will be lists containing
    the losses of the model on the training and validation set at each epoch.
    """

    def __init__(self, model, data, loss_func, learning_rate,
                 is_regression=True, verbose=True, print_every=100):
        """
        Construct a new Solver instance.

        Required arguments:
        - model: A model object conforming to the API described above

        - data: A dictionary of training and validation data with the following:
          'X_train': Training input samples.
          'X_val':   Validation input samples.
          'y_train': Training labels.
          'y_val':   Validation labels.

        - loss_func: Loss function object.
        - learning_rate: Float, learning rate used for gradient descent.

        Optional arguments:
        - verbose: Boolean; if set to false then no output will be printed during
          training.
        - print_every: Integer; training losses will be printed every print_every
          iterations.
        """
        self.model = model
        self.learning_rate = learning_rate
        self.loss_func = loss_func

        # Use an `Optimizer` object to do gradient descent on our model.
        self.opt = Optimizer(model, learning_rate)

        self.is_regression = is_regression
        self.verbose = verbose
        self.print_every = print_every

        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']

        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.best_val_loss = None
        self.best_W = None

        self.train_loss_history = []
        self.val_loss_history = []

    def _step(self):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        """
        model = self.model
        loss_func = self.loss_func
        X_train = self.X_train
        y_train = self.y_train
        opt = self.opt

        model.train()
        model_forward, model_backward = model(X_train)
        loss, loss_grad = loss_func(model_forward, y_train)
        grad = loss_grad*model_backward
        grad = np.mean(grad,0,keepdims=True)
        opt.step(grad.T)


    def check_loss(self, validation=True):
        """
        Check loss of the model on the train/validation data.

        Returns:
        - loss: Averaged loss over the relevant samples.
        """

        X = self.X_val if validation else self.X_train
        y = self.y_val if validation else self.y_train

        model_forward, _ = self.model(X)
        loss, _ = self.loss_func(model_forward, y)

        return loss.mean()

    def train(self, epochs=1000):
        """
        Run optimization to train the model.
        """

        for t in range(epochs):
            # Update the model parameters.
            self._step()

            # Check the performance of the model.
            train_loss = self.check_loss(validation=False)
            val_loss = self.check_loss(validation=True)

            # Record the losses for later inspection.
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)

            if self.verbose and t % self.print_every == 0:
                print('(Epoch %d / %d) train loss: %f; val_loss: %f' % (
                    t, epochs, train_loss, val_loss))

            # Keep track of the best model
            self.update_best_loss(val_loss)

        # At the end of training swap the best params into the model
        self.model.W = self.best_W

    def update_best_loss(self, val_loss):
        # Update the model and best loss if we see improvements.
        if not self.best_val_loss or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_W = self.model.W

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


# 3.数据集CIFAR10分类

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

### 3.2.2Layer Non-Linearities(Sigmoid,Relu)

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

  
