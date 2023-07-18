# 零、学习资源汇总

- dqrobotics的[文档](https://dqroboticsgithubio.readthedocs.io/en/latest/index.html)，[论文](https://arxiv.org/abs/1910.11612)
- vrep(coppeliasim)的[文档](https://www.coppeliarobotics.com/helpFiles/index.html)

- dq-cpp-vrep: [git](https://github.com/dqrobotics/cpp-examples) 

# 一、dqrobotics

## 1. 安装编译

### 1.1 安装

- Matlab安装

  1. 在[网上](https://dqroboticsgithubio.readthedocs.io/en/latest/installation/matlab.html)下载扩展包dqrobotics-YY-MM.mltbx
  2. 在matlab中打开包所在的文件夹，双击这个扩展包，就安装好了

- C++安装

  参照[官网](https://dqroboticsgithubio.readthedocs.io/en/latest/installation/cpp.html#including)

  ```shell
  sudo add-apt-repository ppa:dqrobotics-dev/development
  sudo apt-get update
  sudo apt-get install libdqrobotics					# 安装dq包，包含头：
  	#include <dqrobotics/DQ.h>
  	#include <dqrobotics/robot_modeling/DQ_Kinematics.h>
  	#include <dqrobotics/robot_modeling/DQ_SerialManipulator.h>
  	#include <dqrobotics/utils/DQ_Geometry.h>
  sudo apt-get install libdqrobotics-interface-vrep 	# 安装dq-vrep接口包，包含头：
  	#include<dqrobotics/interfaces/vrep/DQ_VrepInterface.h>
  	#include<dqrobotics/interfaces/vrep/DQ_VrepRobot.h>
  	#include<dqrobotics/interfaces/vrep/robots/LBR4pVrepRobot.h>
  	#include<dqrobotics/interfaces/vrep/robots/YouBotVrepRobot.h>
  ```

### 1.2. C++编译

matlab/python不需要额外写编译文件，c++需要写CMakeLists.txt

- 对于dq包:

  ```cmake
  target_link_libraries(my_binary dqrobotics)
  ```

- 对于dq-vrep接口包:

  ```cmake
  target_link_libraries(my_binary dqrobotics dqrobotics-interface-vrep)
  ```

## 2. 理论

### 2.1 四元数

四元数（Quaternion）是一种数学构造，通常用于表示三维空间中的旋转。四元数由四个实数构成，表示为 (x, y, z, w)。

- x、y、z：表示旋转轴上的向量分量。这些分量定义了旋转轴的方向，指向一个单位向量在三维空间中的 x、y 和 z 方向的分量。
- w：表示旋转角度的余弦一半。它定义了绕旋转轴旋转的角度。四元数表示的旋转角度为原始角度的一半。

#### 2.1.1 轴角转四元数

- 考虑轴$\vec{u}$
  $$
  \vec{u}=(u_x,u_y,u_z)=u_xi+u_yj+u_zk\tag{1}
  $$

- 绕轴旋转$\theta$，表示为四元数
  $$
  q=e^{\frac{\theta}{2}(u_zi+u_yj+u_zk)}=cos\frac{\theta}{2}+(u_xi+u_yj+u_zk)sin\frac{\theta}{2}=cos\frac{\theta}{2}+\vec{u}sin\frac{\theta}{2} \tag{2}
  $$

#### 2.1.2 eigen定义四元数

```c++
#include <Eigen/Geometry>
#include <iostream>

void outputAsMatrix(const Eigen::Quaterniond& q)
{
    // 注意只有单位四元数才表示旋转矩阵，所以要先对四元数单位化
    std::cout << "R=" << std::endl << q.normalized().toRotationMatrix() << std::endl;
}

int main() {
    // 定义一个双精度的四元数
    auto angle = M_PI / 4;
    auto sinA = std::sin(angle / 2);
    auto cosA = std::cos(angle / 2);
    // 方法1：纸上按公式2算好每个值直接写
    Eigen::Quaterniond q(cos(angle / 2), 0, sin(angle / 2), 0);
    // 方法2：更符合公式2的写法
    Eigen::Quaterniond q;
    q.x() = 0 * sinA;
    q.y() = 1 * sinA;
    q.z() = 0 * sinA;
    q.w() = cosA;   
    //方法3：先写轴角然后用eigen转成四元数
    Eigen::AngleAxisd axis_angle(M_PI / 4.0, Eigen::Vector3d::UnitZ());// 定义一个轴角旋转，绕着 z 轴旋转 45 度
    Eigen::Quaterniond quaternion(axis_angle);// 将轴角旋转转换为四元数
    

    // 输出四元数的实部和虚部
    std::cout << "Real part: " << q.w() << std::endl;
    std::cout << "Imaginary parts (x, y, z): " << q.x() << ", " << q.y() << ", " << q.z() << std::endl;
    
    //矩阵的形式输出
    outputAsMatrix(q);
    outputAsMatrix(Eigen::Quaterniond{Eigen::AngleAxisd{angle, Eigen::Vector3d{0, 1, 0}}});

    return 0;
}
```

### 2.2 对偶四元数

- 对偶四元数（Dual Quaternions）使用两个四元数来同时表示平移和旋转
  $$
  u=[q_r,q_d]=q_r+q_d\epsilon \tag{1}
  $$

  - $q_r$：四元数，表示实部real
  - $q_d$：对偶数，表示对偶部dual
  - $\epsilon: \epsilon^2=0,\epsilon \neq0$
  
- 纯旋转的对偶四元数

  即某个单位四元数$q_r$,将它扩展成对偶四元数$u=[q_r,0]$就是一个纯旋转的对偶四元数。

- 纯位移的对偶四元数

  某个位移$t=[t_1,t_2,t_3]$，它所对应的对偶四元数为$u=[1,0.5p]$
  
  - 其中四元数$p=[t_1,t_2,t_3,0]$
  
  - 为什么实部为1？
  
    因为旋转$\theta=0$，所以$q_r=[cos\frac{\theta}{2}+\vec{u}sin\frac{\theta}{2} ]=1$
  
- 刚体运动的对偶四元数
  
  即又有旋转$r$，又有位移$p$，此时:
  $$
  \begin{align}
  u&=[q_r,q_d]=q_r+q_d\epsilon \\
  q_r& = r \\
  q_d& = \frac{1}{2}pr
  \end{align} \tag{2}
  $$

#### 2.2.1 dqrobotics定义对偶四元数

```c++
/* dqrobotics中给了4个别名i_, j_, k_, and E_，分别对应x,y,z轴和\epsilon
下面要定义一个位移为(0.1,0.2,0.3)，绕y轴旋转pi的对偶四元数
*/
// 方法1：
DQ r,p,xd;
r = cos(pi/2) + j_*sin(pi/2); 	// 旋转四元数
p = 0.1*i_ + 0.2*j_ + 0.3*k_; 	// 位移四元数
xd = r + E_*0.5*p*r;			// 应用公式2
// 方法2：
DQ r = DQ(cos(pi/2), 0, sin(pi/2), 0); // 旋转四元数
DQ p = DQ(0, 0.1, 0.2, 0.3);		   // 位移四元数
xd = r + E_*0.5*p*r;				   // 应用公式2
```

#### 2.2.2 dqrobotics和eigen对四元数表示的区别

- dqrobotics:

  如2.2.1所示dqrobotics中四元数不管是内部存储还是复制，它的顺序都是$q=(w,x,y,z)$，即旋转w在最前面。

- eigen:

  - 赋值时是`Quaterniond q(w, x, y, z);`

  - 内部存储是`[x,y,z,w]`

    所以最后输出同样是`[x,y,z,w]`

    



