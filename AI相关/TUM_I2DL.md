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

  - Cross-Entropy loss 多分类

    $E(y,\hat{y};\theta)=-\sum_{i=1}^n\sum_{k=1}^k(y_{ik}\cdot log\hat{y}_{ik})$

  - Loss function: Binary Cross-Entropy (BCE).用于二分类

    $$BCE(y,\hat{y}) =- y \cdot log(\hat y ) - (1- y) \cdot log(1-\hat y) $$

    - $y\in\mathbb{R}$ is the ground truth
    -  $\hat y\in\mathbb{R}$ is the predicted probability of the house being expensive.

- Regression loss

  - L1 loss: $L(y,\hat{y};\theta)=\frac{1}{n}\sum_i^n||y_i-\hat y_i||_1$
  - MSE loss: $L(y,\hat{y};\theta)=\frac{1}{n}\sum_i^n||y_i-\hat{y}_i||_2^2$

## 2) 激活函数Activation Functions

- Sigmoid: $\sigma(x)=\frac{1}{(1+e^{-x})}$
- tanh: $tanh(x)$
- ReLU:$max(0,x)$
- Leaky ReLU: $max(0.1x,x)$

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

  
