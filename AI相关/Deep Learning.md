# 一、Neural Networks and Deep Learning

- Common steps for pre-processing a new dataset:

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

- General Architecture of the learning algorithm

  

# 二、Improving Deep Neural Networks

# 三、Structuring your Machine Learning project

# 四、Convolutional Neural Networks

# 五、Natural Language Processing

