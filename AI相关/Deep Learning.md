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

       ```
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

  ## 3. 



# 二、Improving Deep Neural Networks

# 三、Structuring your Machine Learning project

# 四、Convolutional Neural Networks

# 五、Natural Language Processing

