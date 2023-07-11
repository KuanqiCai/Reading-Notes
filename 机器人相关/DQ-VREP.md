# 零、学习资源汇总

- dqrobotics的[文档](https://dqroboticsgithubio.readthedocs.io/en/latest/index.html)，[论文](https://arxiv.org/abs/1910.11612)
- vrep(coppeliasim)的[文档](https://www.coppeliarobotics.com/helpFiles/index.html)

- dq-cpp-vrep: [git](https://github.com/dqrobotics/cpp-examples) 

# 一、dqrobotics

## 1. 安装

- Matlab安装

  1. 在[网上](https://dqroboticsgithubio.readthedocs.io/en/latest/installation/matlab.html)下载扩展包dqrobotics-YY-MM.mltbx
  2. 在matlab中打开包所在的文件夹，双击这个扩展包，就安装好了

- C++安装

  参照[官网](https://dqroboticsgithubio.readthedocs.io/en/latest/installation/cpp.html#including)

  ```shell
  sudo add-apt-repository ppa:dqrobotics-dev/release
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

## 2. C++编译

matlab/python不需要额外写编译文件，c++需要写CMakeLists.txt

- 对于dq包:

  ```cmake
  target_link_libraries(my_binary dqrobotics)
  ```

- 对于dq-vrep接口包:

  ```cmake
  target_link_libraries(my_binary dqrobotics dqrobotics-interface-vrep)
  ```

## 3. C++使用