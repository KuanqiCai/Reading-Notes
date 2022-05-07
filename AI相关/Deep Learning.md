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

# 二、Improving Deep Neural Networks

# 三、Structuring your Machine Learning project

# 四、Convolutional Neural Networks

# 五、Natural Language Processing

