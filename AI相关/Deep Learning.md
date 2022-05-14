# 一、Neural Networks and Deep Learning

## 1.数据预处理三部曲：

- Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, ...)

  ```python
  # 数据集维度是(m_train,num_px,num_px,3)
  # 3代表RGB3个颜色通道
  m_train = train_set_x_orig.shape[0]
  m_test = test_set_x_orig.shape[0]
  ```

- Reshape the datasets such that each example is now a vector of size (num_px * num_px * 3, 1)

  ```python
  '''
  这里要将数据集的照片从多维转为1维。
  A trick when you want to flatten a matrix X of shape (a,b,c,d) to a matrix X_flatten of shape (b*c*d, a) is to use: 
  	X_flatten = X.reshape(X.shape[0], -1).T
  '''
  train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
  test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
  ```

- "Standardize" the data

  ```python
  # 数据集向量中每一个值都是0到255的RGB值，预处理时要将他们正则化
  train_set_x = train_set_x_flatten / 255.
  test_set_x = test_set_x_flatten / 255.
  ```

## 2.深度学习算法的基本结构

The following Figure explains why **Logistic Regression is actually a very simple Neural Network!**

![](https://github.com/Fernweh-yang/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/deeplearning_1week_1.png?raw=true)



- For one example $x^{(i)}$:
  	$$z^{(i)} = w^T x^{(i)} + b \tag{1}$$
  	$$\hat{y}^{(i)} = a^{(i)} = sigmoid(z^{(i)})\tag{2}$$ 
  	$$ \mathcal{L}(a^{(i)}, y^{(i)}) =  - y^{(i)}  \log(a^{(i)}) - (1-y^{(i)} )  \log(1-a^{(i)})\tag{3}$$

  The cost is then computed by summing over all training examples:
  	$$ J = \frac{1}{m} \sum_{i=1}^m \mathcal{L}(a^{(i)}, y^{(i)})\tag{6}$$

### 1)构建一个Neural Network的3个步骤

  1. Define the model structure (such as number of input features)

     ```python
     def sigmoid(z):
         s = 1/(1+np.exp(-z))
         return s
     ```

  2. Initialize the model's parameters

     ```python
     def initialize_with_zeros(dim):
         w = np.zeros(shape = (dim,1))
         b = float(0) 
         
         #使用断言来确保我要的数据是正确的
         assert(w.shape == (dim, 1)) #w的维度是(dim,1)
         assert(isinstance(b, float) or isinstance(b, int)) #b的类型是float或者是int
         
         return (w , b)
     ```

  3. Loop:

     - Calculate current loss (forward propagation传播)

     - Calculate current gradient (backward propagation)

       ```python
       def propagate(w, b, X, Y):
       	"""
           实现前向和后向传播的成本函数及其梯度。
           参数：
               w  - 权重，大小不等的数组（num_px * num_px * 3，1）
               b  - 偏差，一个标量
               X  - 矩阵类型为（num_px * num_px * 3，训练数量）
               Y  - 真正的“标签”矢量（如果非猫则为0，如果是猫则为1），矩阵维度为(1,训练数据数量)
           返回：
               cost- 逻辑回归的负对数似然成本
               dw  - 相对于w的损失梯度，因此与w相同的形状
               db  - 相对于b的损失梯度，因此与b的形状相同
           """
       	m = X.shape[1]
           
           #正向传播
           A = sigmoid(np.dot(w.T,X) + b) #计算激活值，请参考公式2。
           cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A))) 
           #反向传播
           dw = (1 / m) * np.dot(X, (A - Y).T) 
           db = (1 / m) * np.sum(A - Y) 
           #创建一个字典，把dw和db保存起来。
           grads = {
                       "dw": dw,
                       "db": db
                    }
           return (grads , cost)
       ```

       

     - Update parameters (gradient descent)

       ```python
       def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
           """
           This function optimizes w and b by running a gradient descent algorithm
           
           Arguments:
           w -- weights, a numpy array of size (num_px * num_px * 3, 1)
           b -- bias, a scalar
           X -- data of shape (num_px * num_px * 3, number of examples)
           Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
           num_iterations -- number of iterations of the optimization loop
           learning_rate -- learning rate of the gradient descent update rule
           print_cost -- True to print the loss every 100 steps
           
           Returns:
           params -- dictionary containing the weights w and bias b
           grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
           costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
           """
           
           w = copy.deepcopy(w)
           b = copy.deepcopy(b)
           
           costs = []
           
           for i in range(num_iterations):
               # Cost and gradient calculation 
               grads,cost = propagate(w,b,X,Y)
       
               # Retrieve derivatives from grads
               dw = grads["dw"]
               db = grads["db"]
               
               # update rule (≈ 2 lines of code)
               w=w-learning_rate*dw
               b=b-learning_rate*db
               
               # Record the costs
               if i % 100 == 0:
                   costs.append(cost)
               
                   # Print the cost every 100 training iterations
                   if print_cost:
                       print ("Cost after iteration %i: %f" %(i, cost))
           
           params = {"w": w,
                     "b": b}
           
           grads = {"dw": dw,
                    "db": db}
           
           return params, grads, costs
       ```

       

### 2)然后使用上面算出来的参数w,b来预测新的数据集

  ```python
  def predict(w, b, X):
      '''
      Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
      
      Arguments:
      w -- weights, a numpy array of size (num_px * num_px * 3, 1)
      b -- bias, a scalar
      X -- data of size (num_px * num_px * 3, number of examples)
      
      Returns:
      Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
      '''   
      m = X.shape[1]
      Y_prediction = np.zeros((1, m))
      w = w.reshape(X.shape[0], 1)
      
      # Compute vector "A" predicting the probabilities of a cat being present in the picture
      A=sigmoid(np.dot(w.T,X)+b)
      
      for i in range(A.shape[1]):      
          # Convert probabilities A[0,i] to actual predictions p[0,i]
          if A[0, i] > 0.5 :
              Y_prediction[0,i] =1 
          else:
              Y_prediction[0,i] =0
     
      return Y_prediction
  ```

### 3)最后将上面所有的函数统一到一个函数里去

  ```python
  def model(X_train , Y_train , X_test , Y_test , num_iterations = 2000 , learning_rate = 0.5 , print_cost = False):
      """
      通过调用之前实现的函数来构建逻辑回归模型
      
      参数：
          X_train  - numpy的数组,维度为（num_px * num_px * 3，m_train）的训练集
          Y_train  - numpy的数组,维度为（1，m_train）（矢量）的训练标签集
          X_test   - numpy的数组,维度为（num_px * num_px * 3，m_test）的测试集
          Y_test   - numpy的数组,维度为（1，m_test）的（向量）的测试标签集
          num_iterations  - 表示用于优化参数的迭代次数的超参数
          learning_rate  - 表示optimize（）更新规则中使用的学习速率的超参数
          print_cost  - 设置为true以每100次迭代打印成本
      
      返回：
          d  - 包含有关模型信息的字典。
      """
      w , b = initialize_with_zeros(X_train.shape[0])
      
      parameters , grads , costs = optimize(w , b , X_train , Y_train,num_iterations , learning_rate , print_cost)
      
      #从字典“参数”中检索参数w和b
      w , b = parameters["w"] , parameters["b"]
      
      #预测测试/训练集的例子
      Y_prediction_test = predict(w , b, X_test)
      Y_prediction_train = predict(w , b, X_train)
      
      #打印训练后的准确性
      print("训练集准确性："  , format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100) ,"%")
      print("测试集准确性："  , format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100) ,"%")
      
      d = {
              "costs" : costs,
              "Y_prediction_test" : Y_prediction_test,
              "Y_prediciton_train" : Y_prediction_train,
              "w" : w,
              "b" : b,
              "learning_rate" : learning_rate,
              "num_iterations" : num_iterations }
      return d
  ```

  ## 3. 一个一层隐藏层的神经网络

- 模型图片

  ![](https://github.com/Fernweh-yang/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/Neural%20Network%20model.png?raw=true)

- 对于一个example $x^{(i)}$

  - $$z^{[1] (i)} =  W^{[1]} x^{(i)} + b^{[1]}\tag{1}$$ 
    $$a^{[1] (i)} = \tanh(z^{[1] (i)})\tag{2}$$
    $$z^{[2] (i)} = W^{[2]} a^{[1] (i)} + b^{[2]}\tag{3}$$
    $$\hat{y}^{(i)} = a^{[2] (i)} = \sigma(z^{ [2] (i)})\tag{4}$$
    $$y^{(i)}_{prediction} = \begin{cases} 1 & \mbox{if } a^{[2](i)} > 0.5 \\ 0 & \mbox{otherwise } \end{cases}\tag{5}$$

  - 向量化：
    
    $$Z^{[1]} =  W^{[1]} X + b^{[1]}\tag{1}$$ 
    $$A^{[1]} = \tanh(Z^{[1]})\tag{2}$$
    $$Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}\tag{3}$$
    $$\hat{Y} = A^{[2]} = \sigma(Z^{[2]})\tag{4}$$
    
  - Cost function
  
    $$J = - \frac{1}{m} \sum\limits_{i = 0}^{m} \large\left(\small y^{(i)}\log\left(a^{[2] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[2] (i)}\right)  \large  \right) \small \tag{6}$$

### 1)构建一个神经网络的三个步骤

1. Define the neural network structure ( # of input units,  # of hidden units, etc). 

      ```python
      # 获得每层的尺寸
      def layer_sizes(X, Y):
          """
          Arguments:
          X -- input dataset of shape (input size, number of examples)
          Y -- labels of shape (output size, number of examples)
          
          Returns:
          n_x -- the size of the input layer
          n_h -- the size of the hidden layer 这里根据上面模型设为4
          n_y -- the size of the output layer
          """
          n_x = X.shape[0]
          n_h = 4
          n_y = Y.shape[0]
      
          return (n_x, n_h, n_y)
      ```

  2. Initialize the model's parameters

        ```python
        # 初始化我们要求的模型参数
        def initialize_parameters(n_x, n_h, n_y):
            """
            Argument:
            n_x -- size of the input layer
            n_h -- size of the hidden layer
            n_y -- size of the output layer
            
            Returns:
            params -- python dictionary containing your parameters:
                            W1 -- weight matrix of shape (n_h, n_x)
                            b1 -- bias vector of shape (n_h, 1)
                            W2 -- weight matrix of shape (n_y, n_h)
                            b2 -- bias vector of shape (n_y, 1)
            """    
        
            W1 = np.random.randn(n_h,n_x)*0.01
            b1 = np.zeros(shape=(n_h, 1))
            W2 = np.random.randn(n_y,n_h)*0.01
            b2 = np.zeros(shape=(n_y, 1))
            
        
            parameters = {"W1": W1,
                          "b1": b1,
                          "W2": W2,
                          "b2": b2}
            
            return parameters
        ```

  3. Loop:
        - Implement forward propagation

          ```python
          # 前向传播
          def forward_propagation(X, parameters):
              """
              Argument:
              X -- input data of size (n_x, m)
              parameters -- python dictionary containing your parameters (output of initialization function)
              
              Returns:
              A2 -- The sigmoid output of the second activation
              cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
              """
          
              W1 = parameters["W1"]
              b1 = parameters["b1"]
              W2 = parameters["W2"]
              b2 = parameters["b2"]
          
              Z1 = np.dot(W1 , X) + b1
              A1 = np.tanh(Z1)
              Z2 = np.dot(W2 , A1) + b2
              A2 = sigmoid(Z2)
              
              assert(A2.shape == (1, X.shape[1]))
              
              cache = {"Z1": Z1,
                       "A1": A1,
                       "Z2": Z2,
                       "A2": A2}
              
              return A2, cache
          ```

        - Compute loss

          ```python
          # 计算成本函数
          def compute_cost(A2, Y):
              """
              Computes the cross-entropy cost given in equation (13)
              
              Arguments:
              A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
              Y -- "true" labels vector of shape (1, number of examples)
          
              Returns:
              cost -- cross-entropy cost given equation (13)
              
              """
              
              m = Y.shape[1] 
          
              logprobs = logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
              cost = - np.sum(logprobs) / m
              #   numpy.squeeze() 这个函数的作用是去掉矩阵里维度为1的维度。例如，(1, 5)的矩阵经由np.squeeze处理后变成5；(5, 1, 6)的矩阵经由np.squeeze处理后变成(5, 6)。
              cost = float(np.squeeze(cost))  # makes sure cost is the dimension we expect. 
                                              # E.g., turns [[17]] into 17 
              
              return cost
          ```

        - Implement backward propagation to get the gradients

          ![](https://github.com/Fernweh-yang/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/grad_summary.png?raw=true)

          ```python
          # 计算反向传播是深度学习的难点
          def backward_propagation(parameters, cache, X, Y):
              """
              Implement the backward propagation using the instructions above.
              
              Arguments:
              parameters -- python dictionary containing our parameters 
              cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
              X -- input data of shape (2, number of examples)
              Y -- "true" labels vector of shape (1, number of examples)
              
              Returns:
              grads -- python dictionary containing your gradients with respect to different parameters
              """
              m = X.shape[1]
              W1 = parameters["W1"]
              W2 = parameters["W2"]
              A1 = cache["A1"]
              A2 = cache["A2"]
          
              dZ2= A2 - Y
              dW2 = (1 / m) * np.dot(dZ2, A1.T)
              db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
              dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
              dW1 = (1 / m) * np.dot(dZ1, X.T)
              db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
          
              
              grads = {"dW1": dW1,
                       "db1": db1,
                       "dW2": dW2,
                       "db2": db2}
              
              return grads
          ```

        - Update parameters (gradient descent)

          $\theta=\theta-\alpha\frac{\partial J}{\partial \theta}$

          ```python
          # 更新参数
          def update_parameters(parameters, grads, learning_rate = 1.2):
              """
              Updates parameters using the gradient descent update rule given above
              
              Arguments:
              parameters -- python dictionary containing your parameters 
              grads -- python dictionary containing your gradients 
              
              Returns:
              parameters -- python dictionary containing your updated parameters 
              """
          
              W1,W2 = parameters["W1"],parameters["W2"]
              b1,b2 = parameters["b1"],parameters["b2"]
              dW1,dW2 = grads["dW1"],grads["dW2"]
              db1,db2 = grads["db1"],grads["db2"]
          
              W1 = W1 - learning_rate * dW1
              b1 = b1 - learning_rate * db1
              W2 = W2 - learning_rate * dW2
              b2 = b2 - learning_rate * db2
          
              parameters = {"W1": W1,
                            "b1": b1,
                            "W2": W2,
                            "b2": b2}
              
              return parameters
          ```

### 2)将上面的函数整合到一个函数

```python
def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    np.random.seed(3) #指定随机种子
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    parameters = initialize_parameters(n_x,n_h,n_y)

    for i in range(0, num_iterations):

        A2 , cache = forward_propagation(X,parameters)
        cost = compute_cost(A2,Y)
        grads = backward_propagation(parameters,cache,X,Y)
        parameters = update_parameters(parameters,grads,learning_rate = 1.2)
        
        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters
```

### 3)预测新的数据集

```python
def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
 
    A2 , cache = forward_propagation(X,parameters)
    predictions = np.round(A2)

    return predictions
```

## 4.一个多隐藏层的神经网络

- 一些符号

  - $a^{[L]}$ is the $L^{th}$ layer activation.
  -  $W^{[L]}$ and $b^{[L]}$ are the $L^{th}$ layer parameters.
  - $x^{(i)}$ is the $i^{th}$ training example.
  - $a^{[l]}_i$ denotes the $i^{th}$ entry of the $l^{th}$ layer's activations).
- 通用步骤：

![](https://github.com/Fernweh-yang/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/Deep%20Neural%20Networks.png?raw=true)

For every forward function, there is a corresponding backward function. This is why at every step of your forward module you will be storing some values in a cache. These cached values are useful for computing gradients.

  1. Initialize the parameters for a two-layer network and for an $L$-layer neural network
    
  2. Implement the forward propagation module (shown in purple in the figure below)
    
  3. Compute the loss
    
  4. Implement the backward propagation module (denoted in red in the figure below)
    
  5. Finally, update the parameters

  ### 1) Initialization

- 2-Layer Neural Network

    ```python
    # GRADED FUNCTION: initialize_parameters

    def initialize_parameters(n_x, n_h, n_y):
        """
        Argument:
        n_x -- size of the input layer
        n_h -- size of the hidden layer
        n_y -- size of the output layer

        Returns:
        parameters -- python dictionary containing your parameters:
                        W1 -- weight matrix of shape (n_h, n_x)
                        b1 -- bias vector of shape (n_h, 1)
                        W2 -- weight matrix of shape (n_y, n_h)
                        b2 -- bias vector of shape (n_y, 1)
        """

        np.random.seed(1)
        W1 = np.random.randn(n_h,n_x)*0.01
        b1 = np.zeros((n_h,1))
        W2 = np.random.randn(n_y,n_h)*0.01
        b2 = np.zeros((n_y,1))

        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}

        return parameters    
    ```

- L-Layer Neural Network

  The initialization for a deeper L-layer neural network is more complicated because there are many more weight matrices and bias vectors. When completing the `initialize_parameters_deep` function, you should make sure that your dimensions match between each layer. Recall that $n^{[l]}$ is the number of units in layer $l$. For example, if the size of your input $X$ is $(12288, 209)$ (with $m=209$ examples) then:
  
  |           | Shape of W               | Shape of b       | Activation                                     | Shape of Activation |
  | --------- | ------------------------ | ---------------- | ---------------------------------------------- | ------------------- |
  | Layer 1   | $(n^{[1]},12288)$        | $(n^{[1]},1)$    | $Z^{[1]} = W^{[1]}  X + b^{[1]} $              | $(n^{[1]},209)$     |
  | Layer 2   | $(n^{[2]}, n^{[1]})$     | $(n^{[2]},1)$    | $Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$          | $(n^{[2]}, 209)$    |
  | $\vdots$  | $\vdots$                 | $\vdots$         | $\vdots$                                       | $\vdots$            |
  | Layer L-1 | $(n^{[L-1]}, n^{[L-2]})$ | $(n^{[L-1]}, 1)$ | $Z^{[L-1]} =  W^{[L-1]} A^{[L-2]} + b^{[L-1]}$ | $(n^{[L-1]}, 209)$  |
  | Layer L   | $(n^{[L]}, n^{[L-1]})$   | $(n^{[L]}, 1)$   | $Z^{[L]} =  W^{[L]} A^{[L-1]} + b^{[L]}$       | $(n^{[L]}, 209)$    |
  
  
  Remember that when you compute $W X + b$ in python, it carries out broadcasting. For example, if: 
  
  $$ W = \begin{bmatrix}
      w_{00}  & w_{01} & w_{02} \\
      w_{10}  & w_{11} & w_{12} \\
      w_{20}  & w_{21} & w_{22} 
  \end{bmatrix}\;\;\; X = \begin{bmatrix}
      x_{00}  & x_{01} & x_{02} \\
      x_{10}  & x_{11} & x_{12} \\
      x_{20}  & x_{21} & x_{22} 
  \end{bmatrix} \;\;\; b =\begin{bmatrix}
      b_0  \\
      b_1  \\
      b_2
  \end{bmatrix}\tag{2}$$
  
  Then $WX + b$ will be:
  
  $$ WX + b = \begin{bmatrix}
      (w_{00}x_{00} + w_{01}x_{10} + w_{02}x_{20}) + b_0 & (w_{00}x_{01} + w_{01}x_{11} + w_{02}x_{21}) + b_0 & \cdots \\
      (w_{10}x_{00} + w_{11}x_{10} + w_{12}x_{20}) + b_1 & (w_{10}x_{01} + w_{11}x_{11} + w_{12}x_{21}) + b_1 & \cdots \\
      (w_{20}x_{00} + w_{21}x_{10} + w_{22}x_{20}) + b_2 &  (w_{20}x_{01} + w_{21}x_{11} + w_{22}x_{21}) + b_2 & \cdots
  \end{bmatrix}\tag{3}  $$
  
  ```python
  def initialize_parameters_deep(layer_dims):
      """
      Arguments:
      layer_dims -- python array (list) containing the dimensions of each layer in our network
      
      Returns:
      parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                      Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                      bl -- bias vector of shape (layer_dims[l], 1)
      """
      
      np.random.seed(3)
      parameters = {}
      L = len(layer_dims) # number of layers in the network
  
      for l in range(1, L):
          parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
          parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
  
          
          assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
          assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
  
          
      return parameters
  ```

### 2) Forward Propagation

- Linear_Forward

  $$Z^{[l]} = W^{[l]}A^{[l-1]} +b^{[l]}\tag{4}$$

  where $A^{[0]} = X$. 

  ```python
  def linear_forward(A, W, b):
      """
      Implement the linear part of a layer's forward propagation.
  
      Arguments:
      A -- activations from previous layer (or input data): (size of previous layer, number of examples)
      W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
      b -- bias vector, numpy array of shape (size of the current layer, 1)
  
      Returns:
      Z -- the input of the activation function, also called pre-activation parameter 
      cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
      """
  
      Z = np.dot(W,A)+b
  
      cache = (A, W, b)
      
      return Z, cache
  ```

- Linear_Activation Forward

   There are two activation functions to choose:

  - **Sigmoid**: $\sigma(Z) = \sigma(W A + b) = \frac{1}{ 1 + e^{-(W A + b)}}$.
  -  **ReLU**: The mathematical formula for ReLu is $A = RELU(Z) = max(0, Z)$. 

  ```python
  def linear_activation_forward(A_prev, W, b, activation):
      """
      Implement the forward propagation for the LINEAR->ACTIVATION layer
  
      Arguments:
      A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
      W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
      b -- bias vector, numpy array of shape (size of the current layer, 1)
      activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
  
      Returns:
      A -- the output of the activation function, also called the post-activation value 
      cache -- a python tuple containing "linear_cache" and "activation_cache";
               stored for computing the backward pass efficiently
      """
      
      if activation == "sigmoid":
          Z, linear_cache = linear_forward(A_prev,W,b)
          A, activation_cache = sigmoid(Z)
      
      elif activation == "relu":
          Z, linear_cache = linear_forward(A_prev,W,b)
          A, activation_cache = relu(Z)
  
      cache = (linear_cache, activation_cache)
  
      return A, cache
  ```

- L-layer-model-forward

  传递过程是L-1层 linear->Relu，然后最后一层是Sigmoid

  ![](https://github.com/Fernweh-yang/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/L_layer_model_forward.png?raw=true)

  ```python
  # GRADED FUNCTION: L_model_forward
  
  def L_model_forward(X, parameters):
      """
      Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
      
      Arguments:
      X -- data, numpy array of shape (input size, number of examples)
      parameters -- output of initialize_parameters_deep()
      
      Returns:
      AL -- activation value from the output (last) layer
      caches -- list of caches containing:
                  every cache of linear_activation_forward() (there are L of them, indexed from 0 to L-1)
      """
      caches = []
      A = X
      L = len(parameters) // 2                  # number of layers in the neural network
      
      # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
      # The for loop starts at 1 because layer 0 is the input
      for l in range(1, L):
          A_prev = A 
          A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)],'relu')
          caches.append(cache)
  
      # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
      AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)],'sigmoid')
      caches.append(cache)
           
      return AL, caches
  ```

### 3) Cost Function

$$J=-\frac{1}{m} \sum\limits_{i = 1}^{m} (y^{(i)}\log\left(a^{[L] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[L](i)}\right)) \tag{7}$$

```python
def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    cost = -np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))/m
    
    cost = np.squeeze(cost)    
    
    return cost
```

### 4) Backward Propagation Module

- 与Forward(紫色)的对比

![](https://github.com/Fernweh-yang/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/Back%20vs%20Forward%20.png?raw=true)

- 步骤：

  1. LINEAR backward

  2. LINEAR -> ACTIVATION backward where ACTIVATION computes the derivative of either the ReLU or sigmoid activation

  3. [LINEAR -> RELU] ×× (L-1) -> LINEAR -> SIGMOID backward (whole model)

1. **Linear Backward**

      Suppose you have already calculated the derivative $dZ^{[l]} = \frac{\partial \mathcal{L} }{\partial Z^{[l]}}$. You want to get $(dW^{[l]}, db^{[l]}, dA^{[l-1]})$.

      $$ dW^{[l]} = \frac{\partial \mathcal{J} }{\partial W^{[l]}} = \frac{1}{m} dZ^{[l]} A^{[l-1] T} \tag{8}$$
      $$ db^{[l]} = \frac{\partial \mathcal{J} }{\partial b^{[l]}} = \frac{1}{m} \sum_{i = 1}^{m} dZ^{[l](i)}\tag{9}$$
      $$ dA^{[l-1]} = \frac{\partial \mathcal{L} }{\partial A^{[l-1]}} = W^{[l] T} dZ^{[l]} \tag{10}$$

   ```python
      # GRADED FUNCTION: linear_backward
   
      def linear_backward(dZ, cache):
          """
          Implement the linear portion of backward propagation for a single layer (layer l)
   
          Arguments:
          dZ -- Gradient of the cost with respect to the linear output (of current layer l)
          cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
   
          Returns:
          dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
          dW -- Gradient of the cost with respect to W (current layer l), same shape as W
          db -- Gradient of the cost with respect to b (current layer l), same shape as b
          """
          A_prev, W, b = cache
          m = A_prev.shape[1]
   
          ### START CODE HERE ### (≈ 3 lines of code)
          dW = np.dot(dZ,A_prev.transpose())/m
          db = np.sum(dZ,axis=1,keepdims=True)/m
          dA_prev = np.dot(W.T,dZ)
   
          return dA_prev, dW, db
   ```

2. **Linear-Activation Backward**

   ```python
   def linear_activation_backward(dA, cache, activation):
       """
       Implement the backward propagation for the LINEAR->ACTIVATION layer.
       
       Arguments:
       dA -- post-activation gradient for current layer l 
       cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
       activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
       
       Returns:
       dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
       dW -- Gradient of the cost with respect to W (current layer l), same shape as W
       db -- Gradient of the cost with respect to b (current layer l), same shape as b
       """
       linear_cache, activation_cache = cache
       
       if activation == "relu":
           dZ = relu_backward(dA, activation_cache)
           dA_prev, dW, db = linear_backward(dZ, linear_cache)
           
       elif activation == "sigmoid":
           dZ = sigmoid_backward(dA, activation_cache)
           dA_prev, dW, db = linear_backward(dZ, linear_cache)
   
       return dA_prev, dW, db
   ```

3. **L-Model Backward**

   - Backward pass:

   ![](https://github.com/Fernweh-yang/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/Back%20ward%20pass.png?raw=true)

   - To backpropagate through this network, you know that the output is: 
     $A^{[L]} = \sigma(Z^{[L]})$. Your code thus needs to compute `dAL` $= \frac{\partial \mathcal{L}}{\partial A^{[L]}}$.
     To do so, use this formula (derived using calculus which, again, you don't need in-depth knowledge of!):

     ```python
     dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # derivative of cost with respect to AL
     ```

     You can then use this post-activation gradient `dAL` to keep going backward. As seen in Figure 5, you can now feed in `dAL` into the LINEAR->SIGMOID backward function you implemented (which will use the cached values stored by the L_model_forward function). 

     After that, you will have to use a `for` loop to iterate through all the other layers using the LINEAR->RELU backward function. You should store each dA, dW, and db in the grads dictionary. To do so, use this formula : 

     $$grads["dW" + str(l)] = dW^{[l]}\tag{15} $$

     For example, for $l=3$ this would store $dW^{[l]}$ in `grads["dW3"]`.

   ```python
   def L_model_backward(AL, Y, caches):
       """
       Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
       
       Arguments:
       AL -- probability vector, output of the forward propagation (L_model_forward())
       Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
       caches -- list of caches containing:
                   every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                   the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
       
       Returns:
       grads -- A dictionary with the gradients
                grads["dA" + str(l)] = ... 
                grads["dW" + str(l)] = ...
                grads["db" + str(l)] = ... 
       """
       grads = {}
       L = len(caches) # the number of layers
       m = AL.shape[1]
       Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
       
       # Initializing the backpropagation
       dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
   
       
       # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
       current_cache =  caches[L-1]
       grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
       
       # Loop from l=L-2 to l=0
       for l in reversed(range(L-1)):
           # lth layer: (RELU -> LINEAR) gradients.
           # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
           current_cache = caches[l]
           dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu")
           grads["dA" + str(l)] = dA_prev_temp
           grads["dW" + str(l + 1)] = dW_temp
           grads["db" + str(l + 1)] = db_temp
   
       return grads
   ```

### 5) Update Parameters

$$ W^{[l]} = W^{[l]} - \alpha \text{ } dW^{[l]} \tag{16}$$
$$ b^{[l]} = b^{[l]} - \alpha \text{ } db^{[l]} \tag{17}$$

where $\alpha$ is the learning rate. 

```python
def update_parameters(params, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    params -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    parameters = params.copy()
    L = len(parameters) // 2 

    # Update rule for each parameter. Use a for loop.
    for l in range(L):

        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters
```

## 5.根据4.构建神经网络

### 1) Packages

```python
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v3 import *
from public_tests import *

%matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

%load_ext autoreload
%autoreload 2

np.random.seed(1)
```

### 2) Load and Process the Dataset

```python
'''
    - a training set of `m_train` images labelled as cat (1) or non-cat (0)
    - a test set of `m_test` images labelled as cat and non-cat
    - each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB).
'''
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# Example of a picture
index = 10
plt.imshow(train_x_orig[index])
print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")

# Explore your dataset 
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]
print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))

# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))
```

### 3) Two_Layer Neural Network

#### * Model Architecture

![](https://github.com/Fernweh-yang/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/2_layer_Architecture.png?raw=true)

- The input is a (64,64,3) image which is flattened to a vector of size $(12288,1)$. 
- The corresponding vector: $[x_0,x_1,...,x_{12287}]^T$ is then multiplied by the weight matrix $W^{[1]}$ of size $(n^{[1]}, 12288)$.
- Then, add a bias term and take its relu to get the following vector: $[a_0^{[1]}, a_1^{[1]},..., a_{n^{[1]}-1}^{[1]}]^T$.
- Repeat the same process.
- Multiply the resulting vector by $W^{[2]}$ and add the intercept (bias). 
- Finally, take the sigmoid of the result. If it's greater than 0.5, classify it as a cat.

#### * 2-Layer Model

借助**4.一个多隐藏层神经网络**中创建的函数来实现模型

```python
def initialize_parameters(n_x, n_h, n_y):
    ...
    return parameters 
def linear_activation_forward(A_prev, W, b, activation):
    ...
    return A, cache
def compute_cost(AL, Y):
    ...
    return cost
def linear_activation_backward(dA, cache, activation):
    ...
    return dA_prev, dW, db
def update_parameters(parameters, grads, learning_rate):
    ...
    return parameters
```

- 模型实现：

  ```python
  ### CONSTANTS DEFINING THE MODEL ####
  n_x = 12288     # num_px * num_px * 3
  n_h = 7
  n_y = 1
  layers_dims = (n_x, n_h, n_y)
  learning_rate = 0.0075
  # GRADED FUNCTION: two_layer_model
  
  def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
      """
      Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.
      
      Arguments:
      X -- input data, of shape (n_x, number of examples)
      Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
      layers_dims -- dimensions of the layers (n_x, n_h, n_y)
      num_iterations -- number of iterations of the optimization loop
      learning_rate -- learning rate of the gradient descent update rule
      print_cost -- If set to True, this will print the cost every 100 iterations 
      
      Returns:
      parameters -- a dictionary containing W1, W2, b1, and b2
      """
      
      np.random.seed(1)
      grads = {}
      costs = []                              # to keep track of the cost
      m = X.shape[1]                           # number of examples
      (n_x, n_h, n_y) = layers_dims
      
      # Initialize parameters dictionary, by calling one of the functions you'd previously implemented
      parameters = initialize_parameters(n_x,n_h,n_y);
      
      # Get W1, b1, W2 and b2 from the dictionary parameters.
      W1 = parameters["W1"]
      b1 = parameters["b1"]
      W2 = parameters["W2"]
      b2 = parameters["b2"]
      
      # Loop (gradient descent)
      for i in range(0, num_iterations):
  
          # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1, W2, b2". Output: "A1, cache1, A2, cache2".
          A1,cache1 = linear_activation_forward(X,W1,b1,'relu')
          A2,cache2 = linear_activation_forward(A1,W2,b2,'sigmoid')
          
          # Compute cost
          cost = compute_cost(A2,Y)
          
          # Initializing backward propagation
          dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
          
          # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
          dA1, dW2, db2 = linear_activation_backward(dA2,cache2,'sigmoid')
          dA0, dW1, db1 = linear_activation_backward(dA1,cache1,'relu')
  
          # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
          grads['dW1'] = dW1
          grads['db1'] = db1
          grads['dW2'] = dW2
          grads['db2'] = db2
          
          # Update parameters.
          parameters = update_parameters(parameters,grads,learning_rate)
  
          # Retrieve W1, b1, W2, b2 from parameters
          W1 = parameters["W1"]
          b1 = parameters["b1"]
          W2 = parameters["W2"]
          b2 = parameters["b2"]
          
          # Print the cost every 100 iterations
          if print_cost and i % 100 == 0 or i == num_iterations - 1:
              print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
          if i % 100 == 0 or i == num_iterations:
              costs.append(cost)
  
      return parameters, costs
  
  def plot_costs(costs, learning_rate=0.0075):
      plt.plot(np.squeeze(costs))
      plt.ylabel('cost')
      plt.xlabel('iterations (per hundreds)')
      plt.title("Learning rate =" + str(learning_rate))
      plt.show()
  ```
  
#### * Train the Model

```python
parameters, costs = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)
plot_costs(costs, learning_rate)

predictions_train = predict(train_x, train_y, parameters)

predictions_test = predict(test_x, test_y, parameters)
```



  ### 4) L-layer Neural Network

#### * Model Architecture

![](https://github.com/Fernweh-yang/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/L_layer_Architecture.png?raw=true)

#### * L-layer Model

同样借助**4.一个多隐藏层神经网络**中创建的函数来实现模型

```python
def initialize_parameters_deep(layers_dims):
    ...
    return parameters 
def L_model_forward(X, parameters):
    ...
    return AL, caches
def compute_cost(AL, Y):
    ...
    return cost
def L_model_backward(AL, Y, caches):
    ...
    return grads
def update_parameters(parameters, grads, learning_rate):
    ...
    return parameters
```

- 模型实现

  ```python
  # GRADED FUNCTION: L_layer_model
  
  def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
      """
      Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
      
      Arguments:
      X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
      Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
      layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
      learning_rate -- learning rate of the gradient descent update rule
      num_iterations -- number of iterations of the optimization loop
      print_cost -- if True, it prints the cost every 100 steps
      
      Returns:
      parameters -- parameters learnt by the model. They can then be used to predict.
      """
  
      np.random.seed(1)
      costs = []                         # keep track of cost
      
      # Parameters initialization.
      parameters = initialize_parameters_deep(layers_dims)
      
      # Loop (gradient descent)
      for i in range(0, num_iterations):
  
          # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.  
          AL,caches = L_model_forward(X,parameters)
          
          # Compute cost.
          cost = compute_cost(AL,Y)
      
          # Backward propagation.
          grads = L_model_backward(AL,Y,caches)      
   
          # Update parameters.
          parameters = update_parameters(parameters,grads,learning_rate)
                  
          # Print the cost every 100 iterations
          if print_cost and i % 100 == 0 or i == num_iterations - 1:
              print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
          if i % 100 == 0 or i == num_iterations:
              costs.append(cost)
      
      return parameters, costs
  ```

#### * Train the Model

```python
parameters, costs = L_layer_model(train_x, train_y, layers_dims, num_iterations = 1, print_cost = False)

print("Cost after first iteration: " + str(costs[0]))
L_layer_model_test(L_layer_model)

parameters, costs = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)

pred_train = predict(train_x, train_y, parameters)
pred_test = predict(test_x, test_y, parameters)
```

  

# 二、Improving Deep Neural Networks

# 三、Structuring your Machine Learning project

# 四、Convolutional Neural Networks

# 五、Natural Language Processing

