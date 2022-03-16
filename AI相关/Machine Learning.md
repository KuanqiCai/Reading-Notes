# 一、Introduction

## 1. Supervised Learning监督学习

- 两类问题

  1. Regression problems回归问题

     Given a picture of a person, we have to predict their age on the basis of the given picture

  2. Classification problems分类问题

     Given a patient with a tumor, we have to predict whether the tumor is malignant or benign

- Hypothesis假设h

  <img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/H6qTdZmYEeaagxL7xdFKxA_2f0f671110e8f7446bb2b5b2f75a8874_Screenshot-2016-10-23-20.14.58.png?expiry=1647475200000&hmac=wddnrTefSyJC5CFCVXebNWgz00ra0Ssl-idq70TOXW8" style="zoom:90%;" />

  :比如对于linear Regression线性规划: 

$$
h_\theta(x)=\theta_0+\theta_1x
$$

- cost function:

  We can measure the accuracy of our hypothesis function by using a **cost function**
  $$
  J(\theta_0,\theta_1)=\frac{1}{2m}\sum_{i=1}^m(\hat{y}_i-y_i)^2=\frac{1}{2m}\sum_{i=1}^m(h_\theta(x_i)-y_i)^2
  $$
  目标是最小化这个cost function。
  
- Gradient descent梯度下降

  不仅可用于最小化cost function J，也可用于最小化其他函数。

  - 通用算法：

    repeat until convergence(不断迭代至收敛): 

  $$
  \theta_j:=\theta_j-\alpha\frac{\partial}{\partial\theta_j}J(\theta_0,\theta_1), \quad for\: j=0\: and\:  j=1\\
  这里\alpha是learning\: rate
  $$

  - Subtlety微妙之处：Simultaneous同时发生的。这里两个参数的新值都是基于旧值计算的，两个都算完了再同时代入下一个迭代。
  - learning rate如果
    - 太小，就会导致收敛速度太慢
    - 太大，就可能会导致错过收敛点

## 2. Unsupervised Learning无监督学习

- 两个例子

  1. Clustering:

     Take a collection of 1,000,000 different genes, and find a way to automatically group these genes into groups that are somehow similar or related by different variables, such as lifespan, location, roles, and so on.

  2. Non-Clustering:

     The "Cocktail Party Algorithm", allows you to find structure in a chaotic environment. (i.e. identifying individual voices and music