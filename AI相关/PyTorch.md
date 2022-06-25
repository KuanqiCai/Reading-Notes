# PyTorch安装

- Cuda和cuDNN安装

  - [PyTorch](https://pytorch.org/get-started/locally/)官网查看需要的Cuda版本
  - [Cuda](https://developer.nvidia.com/cuda-toolkit-archive),下载对应的版本
  - [cuDNN](https://developer.nvidia.com/rdp/cudnn-download),下载后解压到cuda[安装的路径](C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3)下

- PyTorch安装

  - [PyTorch](https://pytorch.org/get-started/locally/)官网，查看安装指令
  - 在Conda终端中安装
  - 在需要使用的conda环境下载入安装好的pytorch`conda install pytorch torchvision -c pytorch`

- 查看版本（检查是否安装成功）：

  - PyTorch:

    可以直接查看torch,cuda,cudnn的版本

    ```python
    import torch
    print(torch.__version__) 
    print(torch.version.cuda)
    print(torch.backends.cudnn.version())
    ```

  - Cuda:

    - 命令行：`nvcc --version`

      
    

导入库：

```
import torch
import numpy as np
```

# 一、Tensors

Tensor是一个和numpy的ndarray类似的数据结构，可以运行在GPU或其他硬件加速hardware accelerators上。 In fact, tensors and NumPy arrays can often share the same underlying memory底层内存。

## 1.1 Initializing a Tensor

- **Directly from data**

  ```python
  data = [[1, 2],[3, 4]]
  x_data = torch.tensor(data)
  ```

- **From a NumPy array**

  ```python
  np_array = np.array(data)
  ts_array = torch.from_numpy(np_array)
  ```

  也可以从Tensor转为array

  ```
  np_array_2 = ts_array.numpy() 
  ```

- **From another tensor:**

  如果不特别定义参数，会保留原tensor的shape和datatype

  ```python
  x_ones = torch.ones_like(x_data) # retains the properties of x_data
  print(f"Ones Tensor: \n {x_ones} \n")
  
  x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
  print(f"Random Tensor: \n {x_rand} \n")
  ```

- **With random or constant values:**

  ```python
  shape = (2,3,)
  rand_tensor = torch.rand(shape) # = torch.rand(2,3,)
  ones_tensor = torch.ones(shape)
  zeros_tensor = torch.zeros(shape)
  ```

## 1.2 Attributes of a Tensor

- Tensor attributes describe their shape, datatype, and the device on which they are stored.

  ```python
  tensor = torch.rand(3,4)
  
  print(f"Shape of tensor: {tensor.shape}")
  print(f"Datatype of tensor: {tensor.dtype}")
  print(f"Device tensor is stored on: {tensor.device}")
  ```

  out:

  ```
  Shape of tensor: torch.Size([3, 4])
  Datatype of tensor: torch.float32
  Device tensor is stored on: cpu
  ```

## 1.3 Operations on Tensors

Tensor所有提供的[Operations](https://pytorch.org/docs/stable/torch.html#math-operations)

- **在GPU中计算**

  通常Tensor在cpu中被建立，我们需要将他们移到gpu中保存并计算

  ```python
  # We move our tensor to the GPU if available
  # tensor这里是一个Tensor数据类型的变量名，tensor.to(‘cuda’)只是把tensor这一个变量转到了gpu中保存
  # 如果另一个Tensor数据类型的变量xxx也想转到gpu，需要再输入xxx.to('cuda')
  if torch.cuda.is_available():
    tensor = tensor.to('cuda')
  # 可以再x.to('cpu')转回到cpu存储
  tensor = tensor.to('cpu')
  ```

- **类似numpy.ndarray的索引indexing和切片slicing操作**

  ```python
  tensor = torch.ones(4, 4)
  print('First row: ',tensor[0])
  print('First column: ', tensor[:, 0])
  print('Last column:', tensor[..., -1])
  tensor[:,1] = 0
  print(tensor)
  
  # out
  """
  First row:  tensor([1., 1., 1., 1.])
  First column:  tensor([1., 1., 1., 1.])
  Last column: tensor([1., 1., 1., 1.])
  tensor([[1., 0., 1., 1.],
          [1., 0., 1., 1.],
          [1., 0., 1., 1.],
          [1., 0., 1., 1.]])
  """
  ```

- **将多个Tensor变量整合成一个**

  ```python
  tensor = torch.ones(4, 4)
  t1 = torch.cat([tensor, tensor, tensor], dim=1)
  print(t1)
  
  # out
  """
  tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
          [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
          [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
          [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])
  """
  ```

- **PyTorch中的乘法**：

  torch.matmul()和torch.mul()都会广播

  - 如果维度个数不同，则在维度较少的左边补1，使得维度的个数相同。
  - 各维度的维度大小不同时，如果有维度为1的，直接将该维拉伸至维度相同

  torch.mm()是torch.matmul()的不广播形式

  torch.dot(a,b)是向量相乘，所以输入必须是一维的

  ```python
  #————————————————————————————————————————————————————————
  # This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
  # @运算符：映射torch.matmul()函数，是矩阵乘法
  y1 = tensor @ tensor.T
  y2 = tensor.matmul(tensor.T)
  # 也可以先建一个Tensor大小的变量(y3)，然后将计算结果存在其中
  y3 = torch.rand_like(tensor)
  torch.matmul(tensor, tensor.T, out=y3)
  
  #————————————————————————————————————————————————————————
  # torch.mm()是torch.matmul()的不广播形式
  # 即要求必须满足矩阵相乘维数条件
  y4 = tensor.mm(tensor)
  
  #————————————————————————————————————————————————————————
  # This computes the element-wise product. z1, z2, z3 will have the same value
  # *运算符：映射mul()函数，是2矩阵对应元素element-wise两两相乘
  z1 = tensor * tensor
  z2 = tensor.mul(tensor)
  z3 = torch.rand_like(tensor)
  torch.mul(tensor, tensor, out=z3)
  
  #————————————————————————————————————————————————————————
  #torch.dot(a,b)是向量相乘，所以输入必须是一维的
  a = torch.tensor([2, 3])
  b = torch.tensor([1,2])
  c = torch.dot(a,b)
  ```

- **将Single-element tensors转为python值**

   If you have a one-element tensor, for example by aggregating合计 all values of a tensor into one value, you can convert it to a Python numerical value using `item()`

  ```python
  agg = tensor.sum()
  agg_item = agg.item()
  print(agg_item, type(agg_item))
  
  # out
  """
  12.0 <class 'float'>
  """
  ```

## 1.4 Bridge with Numpy

- 相互转换：

  ```python
  data = [[1, 2],[3, 4]]
  #从numpy转为tensor
  np_array = np.array(data)
  ts_array = torch.from_numpy(np_array)
  #从tensor转为numpy
  np_array_2 = ts_array.numpy() 
  ```

- 相互转换而来的Tensor在cpu和Numpy是共用一个内存underlying memory locations的，所以改变其中一个值，另一个也会改变。

  - 比如上面改变np_array, np_array_2, ts_array任何一个，其他都会跟着变。

    ```python
    # 改变ts_array，np_array和np_array_2也会变
    ts_array.add_(1)
    # 改变np_array, ts_array和np_array_2也会变
    np.add(np_array, 1, out=np_array)
    ```



# 二、Datasets & DataLoaders

- PyTorch提供了两个数据类来帮助模块化处理数据：
  - `torch.utils.data.DataLoader`
    - `DataLoader` wraps an iterable around the `Dataset` to enable easy access to the samples.
  - `torch.utils.data.Datase`
    -  `Dataset` stores the samples and their corresponding labels

- PyTorch也提供各种已经预加载的数据集

  [Image Datasets](https://pytorch.org/vision/stable/datasets.html), [Text Datasets](https://pytorch.org/text/stable/datasets.html), and [Audio Datasets](https://pytorch.org/audio/stable/datasets.html)

## 2.1 Loading a Dataset

- 下面的例子是从TorchVision(一个拥有众多内置图像数据集的模块)中下载Fashion-MNIST数据集。
  - Fashion-MNIST is a dataset of Zalando’s article images consisting of 60,000 training examples and 10,000 test examples. Each example comprises a 28×28 grayscale image and an associated label from one of 10 classes.

    ```python
    import torch
    from torch.utils.data import Dataset
    from torchvision import datasets
    from torchvision.transforms import ToTensor
    import matplotlib.pyplot as plt
    
    training_data = datasets.FashionMNIST(
        # "root" is the path where the train/test data is stored,
        root="data",
        # "train" specifies training or test dataset,
        train=True,
        # "download=True" downloads the data from the internet if it’s not available at root
        download=True,
        # "transform" and "target_transform" specify the feature and label transformations
        transform=ToTensor()
    )
    
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    ```

- `transforms.Compose` creates a series of transformation to prepare the dataset.

  - `transforms.ToTensor` convert `PIL image` or numpy.ndarray $(H \times W\times C)$ in the range [0,255] to a `torch.FloatTensor` of shape $(C \times H \times W)$ in the range [0.0, 1.0].
  - `transforms.Normalize` normalize a tensor image with the provided mean and standard deviation标准偏差.

  ```python
  # Mean and standard deviations have to be sequences (e.g. tuples),hence we add a comma after the values
  transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5,),(0.5,))])
  
  # 相比于上面对数据预处理只用了ToTensor(),这里还进行了图像归一化
  training_data = torchvision.datasets.FashionMNIST(root='../datasets', train=True, download=True, transform=transform)
  
  test_data = torchvision.datasets.FashionMNIST(root='../datasets', train=False, download=True, transform=transform)
  ```

## 2.2 Iterating and Visualizing the Dataset

- `torch.utils.data.Dataloader` takes our training data or test data with parameter `batch_size` and `shuffle`. 

  - The variable `batch_size` defines how many samples per batch to load. 
    - Dataloader可以帮助我们方便的划分barchsize
  - The variable `shuffle=True` makes the data reshuffled重新洗牌 at every epoch.
    - Dataloader可以在一个epoch训练完后，自动的打乱重排每一个batch

  ```python
  # 将数据集training_data和test_data划分各个batch，并得到各个样本的数据和标签
  training_dataloader = DataLoader(training_data, batch_size=8,, shuffle=True)
  test_dataloader = DataLoader(test_data, batch_size=8,, shuffle=True)
  
  classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
             'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
  
  # 迭代training_data数据集,获得各个batch样本的大小，类型，标签
  # 注意都是对dataloader操作，不是对dataset
  for i, item in enumerate(training_dataloader):
      print('Batch {}'.format(i))
      image, label = item
      print(f"Datatype of Image: {type(image)}")
      print(f"Shape of the Image: {image.shape}")
      print(f"Label Values: {label}")
      # 只看第一个batch的
      if i+1 >= 1:
      break
      
  # 用matplotlib来看一些样本图片
  def imshow(img):
      img = img / 2 + 0.5 # unormalize
      npimg = img.numpy()
      plt.imshow(np.transpose(npimg, (1, 2, 0)))
      plt.show()
  # get some random training images
  # 注意都是对dataloader操作，不是对dataset
  dataiter = iter(training_dataloader)
  images, labels = dataiter.next()
  # show images
  imshow(torchvision.utils.make_grid(images))
  # print labels
  print(' '.join('%5s' % classes[labels[j]] for j in range(8)))
  ```

## 2.3 Creating a Custom Dataset for your files

- 一个自定义的数据集类必须包含3个函数

  - `__init__`:initialize the directory containing the images, the annotations file, and both transforms
  - ` __len__ ` :returns the number of samples in our dataset
  - ` __getitem__`:loads and returns a sample from the dataset at the given index

  ```python
  import os
  import pandas as pd
  from torchvision.io import read_image
  
  class CustomImageDataset(Dataset):
      def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
          # their labels are stored separately in a CSV file 
          self.img_labels = pd.read_csv(annotations_file)
          # images are stored in a directory
          self.img_dir = img_dir
          self.transform = transform
          self.target_transform = target_transform
  
      def __len__(self):
          return len(self.img_labels)
  
      def __getitem__(self, idx):
          img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
          image = read_image(img_path)
          label = self.img_labels.iloc[idx, 1]
          if self.transform:
              image = self.transform(image)
          if self.target_transform:
              label = self.target_transform(label)
          return image, label
  ```

  

# 三、 Build Model

- [TORCH.NN库](https://pytorch.org/docs/stable/nn.html#torch-nn)提供了所有用来的建立Neural Network的东西。
  -  [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)是所有PyTorch模块Module的基类
  - A neural network is a module itself that consists of other modules (layers).
    - 这种嵌套的结构使得building and managing complex architectures easily.

- 实例：

  ```python
  import os
  import torch
  from torch import nn
  from torch.utils.data import DataLoader
  from torchvision import datasets, transforms
  #-----------------------Get Device for Training-----------------------
  # 检查是否可以使用GPU
  device = "cuda" if torch.cuda.is_available() else "cpu"
  #-----------------------Define the Class-----------------------
  # 通过继承nn.Module来构建神经网络
  class NeuralNetwork(nn.Module):
      # 在__init__()中初始化神经网络层layers
      def __init__(self):
          # super().__init__让子类不止继承父类的方法，也能继承父类的属性，即父类__init__中的变量值
          super(NeuralNetwork, self).__init__()
          # nn.Flatten()默认参数时，是让输入参数降低一个维度
          self.flatten = nn.Flatten()
          # nn.Sequential()对模块进行封装，这里定义了5层hidden layer
          self.linear_relu_stack = nn.Sequential(
              # Affine layer仿射层：第一个变量是输入维度数量，第二个变量是输出特征数
              # 用于保存weights和biases
              nn.Linear(28*28, 512),
              # 激活层
              nn.ReLU(),
              nn.Linear(512, 512),
              nn.ReLU(),
              nn.Linear(512, 10),
          )
  	# 在forward()中实现对输入数据的操作
      def forward(self, x):
          # 将输入数据展平成1个维度
          x = self.flatten(x)
          # 向前传播
          logits = self.linear_relu_stack(x)
          return logits
  ```

- Model Parameters

  - Many layers inside a neural network are *parameterized*, i.e. have associated weights and biases that are optimized during training.

  - 迭代查看每一层的参数

    ```python
    # 创建我们的神经网络模型的一个实例model，并将它送到gpu
    model = NeuralNetwork().to(device)
    # 模拟一个图片作为输入
    X = torch.rand(1, 28, 28, device=device)
    # 计算输入，得到一个10维的Tensor
    logits = model(X)
    # 用nn.Softmax()来获得预测概率
    pred_probab = nn.Softmax(dim=1)(logits)
    # 返回维度1中最大值的index
    y_pred = pred_probab.argmax(1)
    ```

