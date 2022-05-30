# 一、基本认识

- 一些概念
  - **World**: A world is a 3D description of the properties of robots and of their environment. 不包含controller的代码但会specify机器人所需要的controller的名字。Worlds保存在".wbt"文件内，而".wbt"文件在每一个Webots工程的"worlds"子目录下。
    - 一个world由一个个[node](https://cyberbotics.com/doc/reference/node-chart?tab-language=c)来描述组成
  - **Controller**: A controller is a computer program that controls a robot specified in a world file.可以用c,c++,java,python或者matlab来写。
    - 一个controller可以用于多个机器人，但一个机器人一次只能运行一个controller
  - **Supervisor Controller**: 普通controller的supervisor field字段设置为true。它通常用于执行只能由人类操作员而非机器人执行的操作。In contrast with a regular Robot controller, the Supervisor controller will have access to privileged有特权的 operations. 
- 一些操作

  - shift+鼠标左键：水平移动物体
  - shift+鼠标邮件：旋转物体
  - shift+鼠标左右键：上下移动物体
  - alt+鼠标左键：给物体施加力。注意如果物体（node）质量为0，就无法施加力，所以要先设定质量。
  - [DEF-USE mechanism](https://cyberbotics.com/doc/guide/tutorial-2-modification-of-the-environment?tab-language=c++#:~:text=field%20stands%20for.-,DEF%2DUSE%20Mechanism,-The%20DEF%2DUSE): it allows to define a node in one place and to reuse that definition elsewhere in the scene tree。点NODE 会出来DEF:字段，取个名字，然后在添加node列表里的USE中就会出现我们定义好的node。
    - 注意：只能通过修改原始的DEF来一次性修改所有的node.
  - delete:选中node后+delete删除一个节点

## Node

Nodes用的是VRML97 syntax语法，一系列的node共同构建了world。在Scene Tree栏中，可以看到现在world中所有的nodes。每一个node下都有各种field字段，用于设置该node(物体)的速度、大小、位置、controller等等一切属性。

 ### Solid Node

- Solid Node:A [Solid node](https://cyberbotics.com/doc/reference/solid?tab-language=c++)  represents a **rigid body**刚体, that is a body in which deformation变形 can be neglected忽视。

  Webots只能用于模拟刚体。所以在模拟时，必须要先把不同的entities实体分解成多个刚体。

  - 比如一个桌子，一个轮子。而Soft bodies and articulated铰接 objects are not rigid bodies.但铰接的物体，可以被分为多个solid node
    - a rope绳子, a tire轮胎, a sponge海绵 or an articulated robot arm are not rigid bodies

  - 一个solid node有3个sub_nodes：

    ![](https://github.com/Fernweh-yang/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/Webots/nodes.png?raw=true)

    - children field字段：定义了graphical representation of the Solid node,即长什么样
    - boundingObject：定义了碰撞边界，它的形状可以和shape不同
    - physics：定义了if the object belongs to the dynamical or to the static environment.
      - 要定义pyhsics字段，必须先定义boundingObject字段。

### Nodes related to the graphical

- Shapen node 下的Appearance/PBRAppearance node可以设置物体的表面图案、颜色和纹理。
  - 一些标签
    - baseColor: rgb调色
    - metalness:[0,1]设置金属光泽
    - roughness:设置粗糙度
  - 更倾向于用PBRAppearance，也有很多预定义好的表面
  - PBRAppearance node下的baseColorMap node可以添加自定义的图片纹理texture

# 二、控制

Webots提供的[Nodes和API函数](https://cyberbotics.com/doc/reference/nodes-and-api-functions)。比如下面例子中Robot，DistanceSensor，Motor等Node的变量，函数都可以在这里查到

## 1. 简单的避障

```c++
#include <webots/Robot.hpp>
#include <webots/DistanceSensor.hpp>
#include <webots/Motor.hpp>
// 定义每一个物理step的持续时间，最终间隔是它会乘以WorldInfo Node中'basicTimeStep'这一标签的值
#define TIME_STEP 64
#define MAX_SPEED 6.28
using namespace webots;

// entry point of the controller
// 这里的arg由Robot node的 'controllerArgs'这一标签给予
int main(int argc, char **argv) {
  // create the Robot instance.
  Robot *robot = new Robot();

  // initialize distance sensors
  DistanceSensor *ps[8];
  char psNames[8][4] = {
    "ps0", "ps1", "ps2", "ps3",
    "ps4", "ps5", "ps6", "ps7"
  };
  for (int i = 0; i < 8; i++) {
    ps[i] = robot->getDistanceSensor(psNames[i]);
    ps[i]->enable(TIME_STEP);
  }
	
  // initialize motor
  Motor *leftMotor = robot->getMotor("left wheel motor");
  Motor *rightMotor = robot->getMotor("right wheel motor");
  leftMotor->setPosition(INFINITY);
  rightMotor->setPosition(INFINITY);
  leftMotor->setVelocity(0.0);
  rightMotor->setVelocity(0.0);

  // feedback loop: step simulation until an exit event is received
  while (robot->step(TIME_STEP) != -1) {
    // read sensors outputs
    double psValues[8];
    for (int i = 0; i < 8 ; i++)
      psValues[i] = ps[i]->getValue();

    // detect obstacles
    bool right_obstacle =
      psValues[0] > 80.0 ||
      psValues[1] > 80.0 ||
      psValues[2] > 80.0;
    bool left_obstacle =
      psValues[5] > 80.0 ||
      psValues[6] > 80.0 ||
      psValues[7] > 80.0;

    // initialize motor speeds at 50% of MAX_SPEED.
    double leftSpeed  = 0.5 * MAX_SPEED;
    double rightSpeed = 0.5 * MAX_SPEED;
    // modify speeds according to obstacles
    if (left_obstacle) {
      // turn right
      leftSpeed  = 0.5 * MAX_SPEED;
      rightSpeed = -0.5 * MAX_SPEED;
    }
    else if (right_obstacle) {
      // turn left
      leftSpeed  = -0.5 * MAX_SPEED;
      rightSpeed = 0.5 * MAX_SPEED;
    }
    // write actuators inputs
    leftMotor->setVelocity(leftSpeed);
    rightMotor->setVelocity(rightSpeed);
  }

  delete robot;
  return 0; //EXIT_SUCCESS
}
```

