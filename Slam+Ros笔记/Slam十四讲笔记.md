

[toc]

# 一、预备知识

配套代码：https://github.com/gaoxiang12/slambook2

课后答案参考：https://www.zhihu.com/column/c_1367551490568036352

## 1. Slam全名

- Simultaneous Localization and Mapping 同时定位与地图构建
  - 它是指搭载特定**传感器**的主体，在**没有环境先验信息**的情况下，于**运动过程中**建立**环境**的模型，同时估计自己的**运动**。

## 2. 课后习题

- 线性方程Ax = b的求解，对A和b有哪些要求？（从A的维度和秩角度分析）
  - 若rank(A) < rank(A|b)，则该线性方程为超定方程，无解。（但是可以有最小二乘解
  - 若rank(A) = rank(A|b)，则该线性方程组有解：在此基础上，若rank < A的列数（即变量的个数），则为欠定方程，有无穷解；若rank = A的列数，则为适定方程，具有唯一解。



# 二、初识Slam

## 1. 传感器
### 1.1 传感器分类
- 为了让机器人可以探索一个房间，它需要知道：
  - 定位：我在什么地方
  - 建图：周围环境是什么样
- 传感器可以分为2类
  - 安装于环境中的：  
    比如导轨，二维码等
  - 携带于机器人本体上的：  
    比如轮式编码器、相机、激光传感器、惯性测量单元IMU等
### 1.2 不同的相机特点
- 单目相机Monocular:  
  - 用一个相机的slam称之为单目slam(Monocular Slam)。
  - 由于某个场景scene只在相机的成像平面上留下一个投影，所以天生缺失了一个维度的信息：深度/距离。
  - 为了得到三维结构，就需要让相机运动，然后根据不同地方看物体所形成的视差(disparity)来得到深度信息。
  - 缺点:尺度不确定性Scale Ambiguity
    - 通过平移得到的距离只是一个相对值，我们无法知道那些物体的真实尺寸。
    - 就像看电影，我们虽然能知道一个物体相较另一个物体是大是小，却无法知道他们的真实大小
- 双目相机Stereo：
  - 双目相机由两个单目相机组成，这两相机之间的距离(基线baseline)是已知的。然后根据这个基线来估计每个像素的空间位置
  - 基线距离越大，能够测量得到的物体就越远。
  - 缺点：复杂
    - 其深度量程和精度 受到基线和分辨率限制，所以无人车上双目相机通常很大
    - 视察的计算非常消耗计算资源，需要GPU和FPGA来加速才能实时输出整张图像的距离信息
- 深度相机RGB-D：
  - 可以通过红外结构光或者Time-of-Flight(ToF)原理，来像激光传感器一样，主动向物体发射光并接收返回的光，从而测量距离
  - 由于是根据物理手段测量距离，所以相比双目相机节约了大量计算资源。
  - 缺点：易受干扰
    - RGB-D相机存在测量范围窄、噪声大、视眼小、易受日光干扰、无法测量透射材质等问题。
    - 主要用于室内，室外很难应用。
## 2. SLAM基本框架
![框架](https://img2018.cnblogs.com/blog/1637062/201904/1637062-20190427225259959-530834575.png)

如图所示，主要包括5个步：

1. **传感器信息读取**： 视觉slam中主要为相机图像信息的读取和预处理。还可能有码盘、惯性传感器等。

2. **前端视觉里程计**：（VO: Visual Odometry），估算相邻图像间相机的运动，以及局部地图的样子。VO又称为前端Front End。

   - 视觉里程计：能够通过相邻帧间的图像估计相机运动，并恢复场景的空间结构。

     它和真的里程计一样，只计算相邻时刻的运动，而和过去的信息没有关联。

   - 累积飘逸(Accumulating Drift): 由于两图相间的运动是估算出来的，所以必定有误差，每一时刻的误差都会不断累积传递到下一时刻。由后端和回环消除飘逸。

   - 前端主要研究方向是CV：图像的特征提取与匹配。

3. **后端非线性优化**： 接收不同时刻视觉里程计测量的相机位姿，以及回环检测的信息，对他们进行优化，得到全局一致的轨迹和地图。因为接在VO后，所有称为后端Back End

   - 后端优化主要指处理SLAM过程中的噪声问题，从带有噪声的数据中估计整个系统的状态
   - 最大后验概率估计(Maximum-a-Posteriori)：状态估计的不确定性有多大。
   - 后端的主要研究是：滤波和非线性优化算法。

4. **回环检测**：（Loop Closure Detection）判断机器人是否到达过先前。的位置。如果检测到回环(到达过)，就会把信息提供给后端进行处理。

   - 主要解决位置估计随时间飘逸的问题。即通过某手段让机器人知道它回到原点了，但位置估计还没回到原点，这时将位置估计拉回原点，就消除飘逸了。

5. **建图**：（Mapping）根据估计的轨迹，建立与任务要求对应的地图。

   - 建图没有一个固定的形式和算法，2D栅格、2D拓扑、3D点云、3D网格都可以。

   - 大体可分为**度量地图**和**拓扑地图**：

     - 度量地图Metric Map: 精确表示地图中物体的位置关系，又有2类：

       1. **稀疏**Sparse:  进行一定程度抽象，不表达所有的物体。只选择一些有代表意义的东西作为路标Landmark,不是路标的部分可以忽略。

          定位 稀疏地图就够了。

       2. **稠密**Dense： 建模所有看到的东西。

          导航 需要稠密地图。

     - 拓扑地图Topological Map: 强调地图元素之间的关系。由节点和边组成，只考虑节点间的连通性，不考虑如何从A点到B点。

## 3. SLAM问题数学表述

- 问题：如何用数学描述“一机器人在未知环境中运动”？

  由下面两件事“运动”和“观测”描述。

- 运动：

  可由如下运动方程表示：$x_k=f(x_{k-1},u_k,w_k)$

  - $x_k$: k时刻机器人的位置
  - $u_k$: 运动传感器的读数或输入
  - $w_k$: 噪声
  - $f$: 运动方程

- 观测：

  可由如下观测方程表示：$z_{k,j}=h(y_j,x_k,v_{k,j})\ ,\ (k,j)\in O$

  - $y_j$:路标点
  - $z_{k,j}$:机器人在$x_k$上看到路标点$y_j$时，产生的观测数据
  - $v_{k,j}$：此时的噪声
  - $h$:观测方程
  - $O:$记录哪个时刻观察到什么路标

### 3.1 例子：

描述一个小车在平面中运动

- 运动方程：

  - 小车位姿可由两个轴上的坐标和转角表示：$\mathbf{x}_k=[x_1,x_2,\theta]_k^T$

  - 输入的指令是两个时间间隔之间的 位置和转角变化量：$\mathbf{u}_k=[\Delta x_1,\Delta x_2,\Delta \theta]_k^T$

  - 最终运动方程为：
    $$
    \begin{bmatrix}
       x_1 \\
       x_2 \\
       \theta 
      \end{bmatrix}_k=\begin{bmatrix}
       x_1 \\
       x_2 \\
       \theta 
      \end{bmatrix}_{k-1}+\begin{bmatrix}
       \Delta x_1 \\
       \Delta x_2 \\
       \Delta \theta 
      \end{bmatrix}_k+w_k
    $$

- 观察方程：

  - 通过激光雷达可以观测到2个数据：路标点与小车的 距离$r$和夹角$\phi$

  - 最终观察方程为：
    $$
    \begin{bmatrix}
       r_{k,j} \\
       \phi_{k,j} 
      \end{bmatrix}=\begin{bmatrix}
       \sqrt{(y_{1,j}-x_{1,k})^2+(y_{2,j}-x_{2,k})^2} \\
       arctan\big(\frac{y_{2,j}-x_{2,k}}{y_{1,j}-x_{1,k}}\big) 
      \end{bmatrix}+v
    $$

### 3.2 求解方法：

- 上面这两个方程将SLAM方程变成了一个**状态估计问题**： 如何通过带有噪声的测量数据u，估计内部隐藏着的状态变量x

- 状态估计问题可分为：

  - 线性/非线性系统
  - 高斯/非高斯系统

- 对于简单的线性高斯问题（Linear Gaussian,LG系统）：

  它的无偏最优估计可以由卡尔曼滤波器(Kalman Filter)给出

- 对于复杂的非线性非高斯系统(Non-Linear Non-Gaussian，NLNG系统)：

  有两种方法：

  1. 扩展卡尔曼滤波(Extended Kalman Filter,EKF)
     - 21世纪早期使用的多
     
  2. 非线性优化
     - 现在主流SLAM算法使用图优化(Graph Optimization)为主。
     
# 三、三维空间刚体运动

一个坐标系到另一个坐标系的转换，相差一个欧氏变换（Euclidean Transform）,而欧氏变换又由旋转和平移组成

## 1. 旋转矩阵

Rotation Matrix

- 旋转矩阵属于$SO(n)$特殊正交群(Special Orthogonal Group)
  $$
  SO(n)=\{ R\in\mathbb{R}^{n\times n}|RR^T=I,det(R)=1 \}
  $$

- 某向量$a$在两坐标系下的坐标分别为$a_1,a_2$,他们之间的关系是：
  $$
  a_1=R_{12}a_2+t_{12}
  $$

  - $R_{12}$:坐标系2到坐标系1的旋转。由于向量写在矩阵右侧，所以是从右往左读。
  - $t_{12}$:坐标系1下 坐标系1的原点到坐标系2的原点的向量。

- 由于旋转矩阵是正交矩阵，所以他的逆也是转置描述了一个相反的旋转
  $$
  a=Ra' \\
  a'=R^{-1}a=R^{T}a
  $$
  

## 2. 变换矩阵

Transform Matrix

- 变换矩阵属于$SE(3)$特殊欧氏群(Special Euclidean Group)
  $$
  SE(3)=\{T\in\mathbb{R}^{4\times 4} | R\in SO(3),t\in \mathbb{R}^3 \}
  $$

- 将旋转和平移写入一个矩阵就是变换矩阵
  $$
  \left [\begin{array}{cccc}
  a'\\
  1 \\
  \end{array}\right] = \left [\begin{array}{cccc}
  R & t\\
  0^T & 1 \\
  \end{array}\right]\left [\begin{array}{cccc}
  a \\
  1 \\
  \end{array}\right]=T\left [\begin{array}{cccc}
  a \\
  1 \\
  \end{array}\right]
  $$

- T的逆同样表示一个相反的变换：
  $$
  T^{-1}=\left [\begin{array}{cccc}
  R^T & -R^Tt \\
  0^T & 1 \\
  \end{array}\right]
  $$
  

## 3. 旋转向量

Rotation Vector

- 用旋转矩阵描述旋转有如下两个缺点：

  - SO(3)的旋转矩阵有9个量，但一次旋转只有3个自由度。同样变换矩阵用16个量来表达6个自由度。这种表达方式是冗余redundant的
  - 旋转矩阵自带约束，必须是正交矩阵且行列式为1。这使得求解变得更困难

- 任何一个旋转都可以用一个**旋转轴**和一个**旋转角**来表达：

  - 旋转向量：也叫轴角(Axis-Angle)，方向和旋转轴一致，长度等于旋转角
  - 对于变换矩阵：就可以用一个旋转向量 + 一个平移向量来表示，这样变量数正好是6维

- 罗德里格斯公式(Rodrigues's Formula)：描述旋转向量到旋转矩阵的转换过程
  $$
  \mathbf{R}=cos\theta \mathbf{I}+(1-cos\theta)\mathbf{nn}^T+sin\theta\mathbf{n}^{\wedge}
  $$

  - $\wedge$:向量到反对称矩阵的转换符

  - $\theta$:转角

    可对罗德里格斯公式两边取迹得到：
    $$
    \theta=arccos\frac{tr(\mathbf{R})-1}{2}
    $$

  - $\mathbf{n}$:转轴（向量）

    因为旋转轴上的向量在旋转后不发生改变，所以转轴n也就是旋转矩阵R特征值1对应的特征向量：
    $$
    \mathbf{Rn=n}
    $$

## 4. 欧拉角

Euler Angles

- 无论旋转矩阵还是旋转向量对人类来说都不是很直观，欧拉角用3个分离的转角来把一个旋转分解成3次绕不同轴的旋转。

  - 欧拉角根据先绕什么轴转有各种分类如xyz,zyz,zyx等。
  - 航空中通常用偏航-俯仰-滚转(yaw-pitch-roll)来描述旋转，它等价于ZYX的旋转顺序
    - yaw偏航角：绕Z轴旋转
    - pitch俯仰角：绕旋转之后的Y轴旋转
    - roll滚转角：绕旋转之后的X轴旋转

- 但欧拉角有一个万向锁问题(Gimbal Lock)，也称之为奇异性Singularity

  即如果俯仰角(第二个旋转的角)为+-90°时，第一次旋转和第三次旋转的是同一个轴，这就使得系统丢失了一个自由度

## 5. 四元数

Quaternion

### 5.1 基本概念

- 想用三维向量描述旋转必定带有奇异性，不论是欧拉角还是旋转向量

  这点类似于地球坐标，用经度纬度两个坐标描述，则会在维度+-90°时，经度失去意义，产生奇异性。

- 四元数可以类比于复数的概念，复数的乘法表示复平面上的旋转，比如乘上复数i相当于逆时针把一个复向量旋转90°。

  - 二维：旋转用单位复数来描述

    将复平面的向量逆时针旋转$\theta$角：
    $$
    e^{i\theta}=cos\theta+isin\theta
    $$

  - 三维：旋转用单位四元数来描述
    $$
    \mathbf{q}=q_0+q_1i+q_2j+q_3k
    $$

    - $q_0$：四元数的实部值，代表逆时针旋转角度的余弦值，$q_0=cos\frac{\theta}{2}$

    - $q_1,q_2,q_3$：四元数的虚部值，代表旋转轴的方向，$[q_1\ q_2\ q_3]=[sin\frac{\theta}{2}e_x\ \  sin\frac{\theta}{2}e_y\ \ sin\frac{\theta}{2}e_z]$

    - i,j,k: 四元数的3个虚部，有如下关系：
      $$
      \begin{cases}
      i^2=j^2=k^2=-1\\
      ij=k,ji=-k \\
      jk=i,kj=-i\\
      ki=j,ik=-j
      \end{cases}
      $$

- 如果把i,j,k看作三个坐标轴，如此乘法和外积都和复数一样运算了：
  $$
  \mathbf{q}=[s,\mathbf{v}]^T,\ \ s=q_0\in\mathbb{R},\ \ \mathbf{v}=[q_1,q_2,q_3]^T\in\mathbb{R}^3
  $$

  - 实四元数：如果虚部v为0
  - 虚四元数：如果实部s为0

- 四元数与复数的不同之处：

  - 复数中：乘以i意味着逆时针旋转90°
  - 四元数中：乘以i意味着绕着i轴逆时针旋转了180°

### 5.2 四元数表示旋转

一个三维点$p$ 旋转之后变成$p'$

- 用矩阵描述：$p'=Rp$
- 用四元数描述：
  - 首先把三维点用虚四元数的形式描述：$\mathbf{p}=[0,x,y,z]^T=[0,\mathbf{v}]^T$
  - 经过四元数描述的旋转q:$p'=qpq^{-1}$
    - 这里是四元数乘法，四元数计算规则见书。
  - 最后把$p'$中的虚部取出，就是旋转之后点的坐标

# Eigen库

[官方教程](https://eigen.tuxfamily.org/dox/GettingStarted.html)

## 1.基本使用

### 代码

```c++
#include <iostream>

using namespace std;

#include <ctime>
// Eigen 核心部分
#include <eigen3/Eigen/Core>
// 稠密矩阵的代数运算（逆，特征值等）
#include <eigen3/Eigen/Dense>

using namespace Eigen;

#define MATRIX_SIZE 50

/****************************
* 本程序演示了 Eigen 基本类型的使用
****************************/

int main(int argc, char **argv) {
  // Eigen 中所有向量和矩阵都是Eigen::Matrix，它是一个模板类。它的前三个参数为：数据类型，行，列
  // 声明一个2*3的float矩阵
  Matrix<float, 2, 3> matrix_23;

  // 同时，Eigen 通过 typedef 提供了许多内置类型，不过底层仍是Eigen::Matrix
  // 例如 Vector3d 实质上是 Eigen::Matrix<double, 3, 1>，即三维向量
  Vector3d v_3d;
  // 这是一样的
  Matrix<float, 3, 1> vd_3d;

  // Matrix3d 实质上是 Eigen::Matrix<double, 3, 3>
  Matrix3d matrix_33 = Matrix3d::Zero(); //初始化为零
  // 如果不确定矩阵大小，可以使用动态大小的矩阵
  Matrix<double, Dynamic, Dynamic> matrix_dynamic;
  // 更简单的
  MatrixXd matrix_x;
  // 这种类型还有很多，我们不一一列举

  // 下面是对Eigen阵的操作
  // 输入数据（初始化）
  matrix_23 << 1, 2, 3, 4, 5, 6;
  // 输出
  cout << "matrix 2x3 from 1 to 6: \n" << matrix_23 << endl;

  // 用()访问矩阵中的元素
  cout << "print matrix 2x3: " << endl;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) cout << matrix_23(i, j) << "\t";
    cout << endl;
  }

  // 矩阵和向量相乘（实际上仍是矩阵和矩阵）
  v_3d << 3, 2, 1;
  vd_3d << 4, 5, 6;

  // 但是在Eigen里你不能混合两种不同类型的矩阵，像这样是错的
  // Matrix<double, 2, 1> result_wrong_type = matrix_23 * v_3d;
  // 应该使用cast()将float显式转换为double
  Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;
  cout << "[1,2,3;4,5,6]*[3,2,1]=" << result.transpose() << endl;

  Matrix<float, 2, 1> result2 = matrix_23 * vd_3d;
  cout << "[1,2,3;4,5,6]*[4,5,6]: " << result2.transpose() << endl;

  // 同样你不能搞错矩阵的维度
  // 试着取消下面的注释，看看Eigen会报什么错
  // Eigen::Matrix<double, 2, 3> result_wrong_dimension = matrix_23.cast<double>() * v_3d;

  // 一些矩阵运算
  // 四则运算就不演示了，直接用+-*/即可。
  matrix_33 = Matrix3d::Random();      // 随机数矩阵
  cout << "random matrix: \n" << matrix_33 << endl;
  cout << "transpose: \n" << matrix_33.transpose() << endl;      // 转置
  cout << "sum: " << matrix_33.sum() << endl;            // 各元素和
  cout << "trace: " << matrix_33.trace() << endl;          // 迹
  cout << "times 10: \n" << 10 * matrix_33 << endl;               // 数乘
  cout << "inverse: \n" << matrix_33.inverse() << endl;        // 逆
  cout << "det: " << matrix_33.determinant() << endl;    // 行列式

  // 特征值
  // 实对称矩阵可以保证对角化成功
  // 如有n阶矩阵A，其矩阵元素都为实数，且矩阵A的转置等于其本身aij=aji，则称A为实对称矩阵
  SelfAdjointEigenSolver<Matrix3d> eigen_solver(matrix_33.transpose() * matrix_33);
  cout << "Eigen values = \n" << eigen_solver.eigenvalues() << endl;
  cout << "Eigen vectors = \n" << eigen_solver.eigenvectors() << endl;

  // 解方程
  // 我们求解 matrix_NN * x = v_Nd 这个方程
  // N的大小在前边的宏里定义，它由随机数生成
  // 直接求逆自然是最直接的，但是求逆运算量大

  Matrix<double, MATRIX_SIZE, MATRIX_SIZE> matrix_NN
      = MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
  matrix_NN = matrix_NN * matrix_NN.transpose();  // 保证半正定，半正定矩阵属于实对称矩阵
  Matrix<double, MATRIX_SIZE, 1> v_Nd = MatrixXd::Random(MATRIX_SIZE, 1);

  clock_t time_stt = clock(); // 计时
  // 直接求逆
  Matrix<double, MATRIX_SIZE, 1> x = matrix_NN.inverse() * v_Nd;
  cout << "time of normal inverse is "
       << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
  cout << "x = " << x.transpose() << endl;

  // 通常用矩阵分解来求，例如QR分解，速度会快很多
  time_stt = clock();
  x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
  cout << "time of Qr decomposition is "
       << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
  cout << "x = " << x.transpose() << endl;

  // 对于正定矩阵，还可以用cholesky分解来解方程
  // 半正定矩阵：给定一个大小为nXn的实对称矩阵A，若对于任意长度为n的向量x,有xTAx>=0恒成立，则矩阵A是一个半正定矩阵。
  // 正定矩阵：给定一个大小为nXn的实对称矩阵A，若对于任意长度为n的非零向量x,有xTAx>0恒成立，则矩阵A是一个正定矩阵。
  // 半正定矩阵包括了正定矩阵。
  time_stt = clock();
  x = matrix_NN.ldlt().solve(v_Nd);
  cout << "time of ldlt decomposition is "
       << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
  cout << "x = " << x.transpose() << endl;

  return 0;
}
```

### 编译运行

CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 2.8)
project(useEigen)

set(CMAKE_BUILD_TYPE "Release")
# CMAKE_CXX_FLAGS是CMake传给c++编译器的编译选项，就好比g++ -03
# CMAKE_CXX_FLAGS_DEBUG,是除了CMAKE_CXX_FLAGS外，在Debug配置下，额外的参数
# CMAKE_CXX_FLAGS_RELEASE，是除了CMAKE_CXX_FLAGS外，在Release配置下，额外的参数
set(CMAKE_CXX_FLAGS "-O3")

# 添加Eigen头文件
# Eigen库只有头文件没有库文件，所以不需要target_link_libraries.
# 这么做的坏处：如果把Eigen安装在了不同位置，就必须要手动修改头文件目录。
include_directories("/usr/include/eigen3")
add_executable(eigenMatrix eigenMatrix.cpp)
```

编译运行

```
mkdir build
cd build 
cmake ..
make 
./eigenMatrix
```

## 2.几何模块基本使用

[官方教程](https://eigen.tuxfamily.org/dox/group__TutorialGeometry.html)

### 代码

```c++
#include <iostream>
#include <cmath>

using namespace std;

#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace Eigen;

// 本程序演示了 Eigen 几何模块的使用方法

int main(int argc, char **argv) {

  // Eigen/Geometry 模块提供了各种旋转和平移的表示
  // 3D 旋转矩阵直接使用 Matrix3d 或 Matrix3f，d是double，f是float
  // Identity()用单位矩阵对新变量进行初始化
  Matrix3d rotation_matrix = Matrix3d::Identity();
  // 旋转向量使用 AngleAxis, 它底层不直接是Matrix，但运算可以当作矩阵（因为重载了运算符）
  AngleAxisd rotation_vector(M_PI / 4, Vector3d(0, 0, 1));     //沿 Z 轴旋转 45 度
  cout.precision(3);
  cout << "rotation matrix =\n" << rotation_vector.matrix() << endl;   //用matrix()转换成矩阵
  // 也可以直接赋值
  rotation_matrix = rotation_vector.toRotationMatrix();
  // 用 AngleAxis 可以进行坐标变换
  Vector3d v(1, 0, 0);
  Vector3d v_rotated = rotation_vector * v;
  //vector.transpose()输出向量/矩阵的转置
  cout << "(1,0,0) after rotation (by angle axis) = " << v_rotated.transpose() << endl;
  // 或者用旋转矩阵
  v_rotated = rotation_matrix * v;
  cout << "(1,0,0) after rotation (by matrix) = " << v_rotated.transpose() << endl;

  // 欧拉角: 可以将旋转矩阵直接转换成欧拉角
  Vector3d euler_angles = rotation_matrix.eulerAngles(2, 1, 0); // ZYX顺序，即yaw-pitch-roll顺序
  cout << "yaw pitch roll = " << euler_angles.transpose() << endl;

  // 欧氏变换矩阵使用 Eigen::Isometry
  Isometry3d T = Isometry3d::Identity();                // 虽然称为3d，实质上是4＊4的矩阵
  T.rotate(rotation_vector);                                     // 按照rotation_vector进行旋转
  T.pretranslate(Vector3d(1, 3, 4));                     // 把平移向量设成(1,3,4)
  cout << "Transform matrix = \n" << T.matrix() << endl;

  // 用变换矩阵进行坐标变换
  Vector3d v_transformed = T * v;                              // 相当于R*v+t
  cout << "v tranformed = " << v_transformed.transpose() << endl;

  // 对于仿射和射影变换，使用 Eigen::Affine3d 和 Eigen::Projective3d 即可，略

  // 四元数
  // 可以直接把AngleAxis赋值给四元数，反之亦然
  Quaterniond q = Quaterniond(rotation_vector);
  // coeffs()函数返回四元素的四个参数。
  cout << "quaternion from rotation vector = " << q.coeffs().transpose()
       << endl;   // 请注意coeffs的顺序是(x,y,z,w),w为实部，前三者为虚部
  // 也可以把旋转矩阵赋给它
  q = Quaterniond(rotation_matrix);
  cout << "quaternion from rotation matrix = " << q.coeffs().transpose() << endl;
  // 使用四元数旋转一个向量，使用重载的乘法即可
  v_rotated = q * v; // 注意数学上是qvq^{-1}
  cout << "(1,0,0) after rotation = " << v_rotated.transpose() << endl;
  // 用常规向量乘法表示，则应该如下计算
  cout << "should be equal to " << (q * Quaterniond(0, 1, 0, 0) * q.inverse()).coeffs().transpose() << endl;

  return 0;
}
```

### 编译运行

```cmake
cmake_minimum_required( VERSION 2.8 )
project( geometry )

# 添加Eigen头文件
include_directories( "/usr/include/eigen3" )

add_executable(eigenGeometry eigenGeometry.cpp)
```

## 3. 一个例子

### 问题描述

设有小萝卜1号和小萝卜2号位于世界坐标系中。记世界坐标系为W，小萝卜们的坐标系为R1和R2。

小萝卜1号的位姿为q1=[0.35, 0.2, 0.3, 0.1]T，t1=[0.3, 0.1, 0.1]T。

小萝卜2号的位姿为q2=[-0.5, 0.4, -0.1, 0.2]T，t2=[-0.1, 0.5, 0.3]T。

这里的q,t即T，表示世界坐标系到相机坐标系的变换关系。

现在小萝卜1号看到某个点在自身的坐标系下坐标为PR1=[0.4, 0, 0.2]T，求该向量在小萝卜2号坐标系下的坐标。

### 代码

```c++
#include <iostream>
#include <vector>
#include <algorithm>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

using namespace std;
using namespace Eigen;

int main(int argc, char** argv){
    Quaterniond q1(0.35, 0.2, 0.3, 0.1), q2(-0.5, 0.4, -0.1, 0.2);
    // 归一化2个四元数
    q1.normalize();
    q2.normalize();
    Vector3d t1(0.3, 0.1, 0.1), t2(-0.3, 0.5, 0.3);
    Vector3d p1(0.5, 0, 0.2);
    
    // 欧式变换矩阵
    Isometry3d T1w(q1),T2w(q2);
    // 设置变换矩阵中的位移t
    T1w.pretranslate(t1);
    T2w.pretranslate(t2);

    // 先从1转世界，再从世界转2坐标系
    Vector3d p2 = T2w*T1w.inverse()*p1;
    cout << endl << p2.transpose() <<endl;
    return 0;
}
```



# Sophus库

## 1.安装

1. 安装依赖库fmt

   ```shell
   # 最新版的fmt会导致Sophus编译报错，所以安装8.0.0版本
   # 首先删除之前安装的版本，一般make install完了目录下都会有一个install_manifest.txt的文件记录安装的所有内容，通过如下命令来删除：
   xargs rm < install_manifest.txt
   git clone -b 8.0.0 git@github.com:fmtlib/fmt.git
   cd fmt
   mkdir build 
   cd build
   cmake ..
   make
   sudo make install
   ```

2. 安装Sophus库

   ```shell
   git clone git@github.com:strasdat/Sophus.git
   cd Sophus
   mkdir build
   cd build
   cmake ..
   make
   sudo make install
   ```

   

## 2. 基本使用

### 代码

```c++
#include <iostream>
#include <cmath>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include "sophus/se3.hpp"

using namespace std;
using namespace Eigen;

/// 本程序演示sophus的基本用法

int main(int argc, char **argv) {

  // 沿Z轴转90度的旋转矩阵
  Matrix3d R = AngleAxisd(M_PI / 2, Vector3d(0, 0, 1)).toRotationMatrix();
  // 或者四元数
  Quaterniond q(R);
  Sophus::SO3d SO3_R(R);              // Sophus::SO3d可以直接从旋转矩阵构造
  Sophus::SO3d SO3_q(q);              // 也可以通过四元数构造
  // 二者是等价的
  cout << "SO(3) from matrix:\n" << SO3_R.matrix() << endl;
  cout << "SO(3) from quaternion:\n" << SO3_q.matrix() << endl;
  cout << "they are equal" << endl;

  // 使用对数映射获得它的李代数
  Vector3d so3 = SO3_R.log();
  cout << "so3 = " << so3.transpose() << endl;
  // hat 为向量到反对称矩阵，即向上的大于号：从向量到矩阵
  cout << "so3 hat=\n" << Sophus::SO3d::hat(so3) << endl;
  // 相对的，vee为反对称到向量，即向下的大于号：从矩阵到向量
  cout << "so3 hat vee= " << Sophus::SO3d::vee(Sophus::SO3d::hat(so3)).transpose() << endl;

  // 增量扰动模型的更新
  // 书p83,85。dR*R在李代数的表示
  Vector3d update_so3(1e-4, 0, 0); //假设更新量为这么多
  Sophus::SO3d SO3_updated = Sophus::SO3d::exp(update_so3) * SO3_R;
  cout << "SO3 updated = \n" << SO3_updated.matrix() << endl;

  cout << "*******************************" << endl;
  // 对SE(3)操作大同小异
  Vector3d t(1, 0, 0);           // 沿X轴平移1
  Sophus::SE3d SE3_Rt(R, t);           // 从R,t构造SE(3)
  Sophus::SE3d SE3_qt(q, t);            // 从q,t构造SE(3)
  cout << "SE3 from R,t= \n" << SE3_Rt.matrix() << endl;
  cout << "SE3 from q,t= \n" << SE3_qt.matrix() << endl;
  // 李代数se(3) 是一个六维向量，方便起见先typedef一下
  typedef Eigen::Matrix<double, 6, 1> Vector6d;
  Vector6d se3 = SE3_Rt.log();
  cout << "se3 = " << se3.transpose() << endl;
  // 观察输出，会发现在Sophus中，se(3)的平移在前，旋转在后.
  // 同样的，有hat和vee两个算符
  cout << "se3 hat = \n" << Sophus::SE3d::hat(se3) << endl;
  cout << "se3 hat vee = " << Sophus::SE3d::vee(Sophus::SE3d::hat(se3)).transpose() << endl;

  // 最后，演示一下更新
  Vector6d update_se3; //更新量
  update_se3.setZero();
  // 李代数第一个量设置为0.0001
  update_se3(0, 0) = 1e-4;
  Sophus::SE3d SE3_updated = Sophus::SE3d::exp(update_se3) * SE3_Rt;
  cout << "SE3 updated = " << endl << SE3_updated.matrix() << endl;

  return 0;
}
```

### 编译运行

CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.0)
project(useSophus)

# 为使用 sophus，需要使用find_package命令找到它
find_package(Sophus REQUIRED)

# Eigen
include_directories("/usr/include/eigen3")
add_executable(useSophus useSophus.cpp)
target_link_libraries(useSophus Sophus::Sophus)
```

## 3. 例子：评估轨迹的误差

### 代码

```c++
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <pangolin/pangolin.h>
#include <sophus/se3.hpp>

using namespace Sophus;
using namespace std;

string groundtruth_file = "./example/groundtruth.txt";
string estimated_file = "./example/estimated.txt";

typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;

void DrawTrajectory(const TrajectoryType &gt, const TrajectoryType &esti);

TrajectoryType ReadTrajectory(const string &path);

int main(int argc, char **argv) {
  TrajectoryType groundtruth = ReadTrajectory(groundtruth_file);
  TrajectoryType estimated = ReadTrajectory(estimated_file);
  assert(!groundtruth.empty() && !estimated.empty());
  assert(groundtruth.size() == estimated.size());

  // compute rmse
  double rmse = 0;
  for (size_t i = 0; i < estimated.size(); i++) {
    Sophus::SE3d p1 = estimated[i], p2 = groundtruth[i];
    double error = (p2.inverse() * p1).log().norm();
    rmse += error * error;
  }
  rmse = rmse / double(estimated.size());
  rmse = sqrt(rmse);
  cout << "RMSE = " << rmse << endl;

  DrawTrajectory(groundtruth, estimated);
  return 0;
}

TrajectoryType ReadTrajectory(const string &path) {
  ifstream fin(path);
  TrajectoryType trajectory;
  if (!fin) {
    cerr << "trajectory " << path << " not found." << endl;
    return trajectory;
  }

  while (!fin.eof()) {
    double time, tx, ty, tz, qx, qy, qz, qw;
    fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
    Sophus::SE3d p1(Eigen::Quaterniond(qw, qx, qy, qz), Eigen::Vector3d(tx, ty, tz));
    trajectory.push_back(p1);
  }
  return trajectory;
}

void DrawTrajectory(const TrajectoryType &gt, const TrajectoryType &esti) {
  // create pangolin window and plot the trajectory
  pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
      pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
  );

  pangolin::View &d_cam = pangolin::CreateDisplay()
      .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
      .SetHandler(new pangolin::Handler3D(s_cam));


  while (pangolin::ShouldQuit() == false) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    d_cam.Activate(s_cam);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    glLineWidth(2);
    for (size_t i = 0; i < gt.size() - 1; i++) {
      glColor3f(0.0f, 0.0f, 1.0f);  // blue for ground truth
      glBegin(GL_LINES);
      auto p1 = gt[i], p2 = gt[i + 1];
      glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
      glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
      glEnd();
    }

    for (size_t i = 0; i < esti.size() - 1; i++) {
      glColor3f(1.0f, 0.0f, 0.0f);  // red for estimated
      glBegin(GL_LINES);
      auto p1 = esti[i], p2 = esti[i + 1];
      glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
      glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
      glEnd();
    }
    pangolin::FinishFrame();
    usleep(5000);   // sleep 5 ms
  }

}
```

### 编译运行

```cmake
find_package(Pangolin REQUIRED)
find_package(Sophus REQUIRED)

include_directories(${Pangolin_INCLUDE_DIRS})
add_executable(trajectoryError trajectoryError.cpp)
target_link_libraries(trajectoryError ${Pangolin_LIBRARIES})
target_link_libraries(trajectoryError Sophus::Sophus)
```



#  Pangolin库

## 1. 安装库

1. 从[官方代码库](https://github.com/stevenlovegrove/Pangolin)git下代码来。

2. 安装依赖:`sudo apt-get install libglew-dev`

3. 编译库

   ```
   cd [path-to-pangolin]
   mkdir build
   cd build
   cmake ..
   make 
   sudo make install 
   sudo ldconfig
   ```

## 2. 示例代码

### 画一个已知的轨迹

#### 代码

```c++
#include <pangolin/pangolin.h>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <unistd.h>

// 本例演示了如何画出一个预先存储的轨迹

using namespace std;
using namespace Eigen;

// path to trajectory file
// 导入轨迹
string trajectory_file = "./examples/trajectory.txt";

void DrawTrajectory(vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>>);

int main(int argc, char **argv) {

  vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> poses;
  ifstream fin(trajectory_file);
  if (!fin) {
    cout << "cannot find trajectory file at " << trajectory_file << endl;
    return 1;
  }

  while (!fin.eof()) {
    double time, tx, ty, tz, qx, qy, qz, qw;
    fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
    Isometry3d Twr(Quaterniond(qw, qx, qy, qz));
    Twr.pretranslate(Vector3d(tx, ty, tz));
    poses.push_back(Twr);
  }
  cout << "read total " << poses.size() << " pose entries" << endl;

  // draw trajectory in pangolin
  DrawTrajectory(poses);
  return 0;
}

/*******************************************************************************************/
void DrawTrajectory(vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> poses) {
  // create pangolin window and plot the trajectory
  pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  pangolin::OpenGlRenderState s_cam(
    pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
    pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
  );

  pangolin::View &d_cam = pangolin::CreateDisplay()
    .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
    .SetHandler(new pangolin::Handler3D(s_cam));

  while (pangolin::ShouldQuit() == false) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    d_cam.Activate(s_cam);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glLineWidth(2);
    for (size_t i = 0; i < poses.size(); i++) {
      // 画每个位姿的三个坐标轴
      Vector3d Ow = poses[i].translation();
      Vector3d Xw = poses[i] * (0.1 * Vector3d(1, 0, 0));
      Vector3d Yw = poses[i] * (0.1 * Vector3d(0, 1, 0));
      Vector3d Zw = poses[i] * (0.1 * Vector3d(0, 0, 1));
      glBegin(GL_LINES);
      glColor3f(1.0, 0.0, 0.0);
      glVertex3d(Ow[0], Ow[1], Ow[2]);
      glVertex3d(Xw[0], Xw[1], Xw[2]);
      glColor3f(0.0, 1.0, 0.0);
      glVertex3d(Ow[0], Ow[1], Ow[2]);
      glVertex3d(Yw[0], Yw[1], Yw[2]);
      glColor3f(0.0, 0.0, 1.0);
      glVertex3d(Ow[0], Ow[1], Ow[2]);
      glVertex3d(Zw[0], Zw[1], Zw[2]);
      glEnd();
    }
    // 画出连线
    for (size_t i = 0; i < poses.size(); i++) {
      glColor3f(0.0, 0.0, 0.0);
      glBegin(GL_LINES);
      auto p1 = poses[i], p2 = poses[i + 1];
      glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
      glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
      glEnd();
    }
    pangolin::FinishFrame();
    usleep(5000);   // sleep 5 ms
  }
}
```

#### 编译运行

CMakeLists.txt

```cmake
include_directories("/usr/include/eigen3")

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})
add_executable(plotTrajectory plotTrajectory.cpp)
target_link_libraries(plotTrajectory ${Pangolin_LIBRARIES})
```

命令行输入

```
mkdir build
cd build
cmake ..
make 

./plotTrajectory
```

### 可视化相机位姿的各种表达方式

#### 代码

```c++
#include <iostream>
#include <iomanip>

using namespace std;

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

using namespace Eigen;

#include <pangolin/pangolin.h>

struct RotationMatrix {
  Matrix3d matrix = Matrix3d::Identity();
};

ostream &operator<<(ostream &out, const RotationMatrix &r) {
  out.setf(ios::fixed);
  Matrix3d matrix = r.matrix;
  out << '=';
  out << "[" << setprecision(2) << matrix(0, 0) << "," << matrix(0, 1) << "," << matrix(0, 2) << "],"
      << "[" << matrix(1, 0) << "," << matrix(1, 1) << "," << matrix(1, 2) << "],"
      << "[" << matrix(2, 0) << "," << matrix(2, 1) << "," << matrix(2, 2) << "]";
  return out;
}

istream &operator>>(istream &in, RotationMatrix &r) {
  return in;
}

struct TranslationVector {
  Vector3d trans = Vector3d(0, 0, 0);
};

ostream &operator<<(ostream &out, const TranslationVector &t) {
  out << "=[" << t.trans(0) << ',' << t.trans(1) << ',' << t.trans(2) << "]";
  return out;
}

istream &operator>>(istream &in, TranslationVector &t) {
  return in;
}

struct QuaternionDraw {
  Quaterniond q;
};

ostream &operator<<(ostream &out, const QuaternionDraw quat) {
  auto c = quat.q.coeffs();
  out << "=[" << c[0] << "," << c[1] << "," << c[2] << "," << c[3] << "]";
  return out;
}

istream &operator>>(istream &in, const QuaternionDraw quat) {
  return in;
}

int main(int argc, char **argv) {
  pangolin::CreateWindowAndBind("visualize geometry", 1000, 600);
  glEnable(GL_DEPTH_TEST);
  pangolin::OpenGlRenderState s_cam(
    pangolin::ProjectionMatrix(1000, 600, 420, 420, 500, 300, 0.1, 1000),
    pangolin::ModelViewLookAt(3, 3, 3, 0, 0, 0, pangolin::AxisY)
  );

  const int UI_WIDTH = 500;

  pangolin::View &d_cam = pangolin::CreateDisplay().
    SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, -1000.0f / 600.0f).
    SetHandler(new pangolin::Handler3D(s_cam));

  // ui
  pangolin::Var<RotationMatrix> rotation_matrix("ui.R", RotationMatrix());
  pangolin::Var<TranslationVector> translation_vector("ui.t", TranslationVector());
  pangolin::Var<TranslationVector> euler_angles("ui.rpy", TranslationVector());
  pangolin::Var<QuaternionDraw> quaternion("ui.q", QuaternionDraw());
  pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));

  while (!pangolin::ShouldQuit()) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    d_cam.Activate(s_cam);

    pangolin::OpenGlMatrix matrix = s_cam.GetModelViewMatrix();
    Matrix<double, 4, 4> m = matrix;

    RotationMatrix R;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        R.matrix(i, j) = m(j, i);
    rotation_matrix = R;

    TranslationVector t;
    t.trans = Vector3d(m(0, 3), m(1, 3), m(2, 3));
    t.trans = -R.matrix * t.trans;
    translation_vector = t;

    TranslationVector euler;
    euler.trans = R.matrix.eulerAngles(2, 1, 0);
    euler_angles = euler;

    QuaternionDraw quat;
    quat.q = Quaterniond(R.matrix);
    quaternion = quat;

    glColor3f(1.0, 1.0, 1.0);

    pangolin::glDrawColouredCube();
    // draw the original axis
    glLineWidth(3);
    glColor3f(0.8f, 0.f, 0.f);
    glBegin(GL_LINES);
    glVertex3f(0, 0, 0);
    glVertex3f(10, 0, 0);
    glColor3f(0.f, 0.8f, 0.f);
    glVertex3f(0, 0, 0);
    glVertex3f(0, 10, 0);
    glColor3f(0.2f, 0.2f, 1.f);
    glVertex3f(0, 0, 0);
    glVertex3f(0, 0, 10);
    glEnd();

    pangolin::FinishFrame();
  }
}
```

#### 编译运行

CMakeLists.txt

```c++
cmake_minimum_required( VERSION 2.8 )
project( visualizeGeometry )

set(CMAKE_CXX_FLAGS "-std=c++17")

# 添加Eigen头文件
include_directories( "/usr/include/eigen3" )

# 添加Pangolin依赖
find_package( Pangolin )
include_directories( ${Pangolin_INCLUDE_DIRS} )

add_executable( visualizeGeometry visualizeGeometry.cpp )
target_link_libraries( visualizeGeometry ${Pangolin_LIBRARIES} )
```

