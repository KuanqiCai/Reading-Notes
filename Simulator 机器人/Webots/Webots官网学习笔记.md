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
- 坐标轴：
  - 红：X
  - 绿：y
  - 蓝：z


## Node

[Node所有的分类](https://cyberbotics.com/doc/reference/node-chart?tab-language=c++)

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

- Shapen node 下的Appearance/PBRAppearance node可以设置物体的表面图案、颜色和纹理。
  - 一些标签
    - baseColor: rgb调色
    - metalness:[0,1]设置金属光泽
    - roughness:设置粗糙度
  - 更倾向于用PBRAppearance，也有很多预定义好的表面
  - PBRAppearance node下的baseColorMap node可以添加自定义的图片纹理texture
- 大多数的sensors和actuators同时是Solid和Device Node

### Robot nodel

- Robot nodel是一个相互联系的solide nodes组成的树结构，这个树的root node是robot
  - 这些solid node通过joint nodes连接，机器人中常用HingeJoint node来连接两个Solid Nodes
  - device node应该是任意robot/solid/joint node的一个直接child node
  - 一个Joint node可以通过添加PositionSensor node或在device字段添加motor node来monitor监视/actuate驱动
  - 在Humanoid robot中，通常robot node(root node)是它的chest胸部

### Supervisor

[Supervisor](https://cyberbotics.com/doc/reference/supervisor?tab-language=c++)是一个特殊的Robot nodel(只需要supervisor field设置为true)。

它可以通过添加或删除nodes来改变环境。

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

## 2.一个四轮机器人

[教程](https://cyberbotics.com/doc/guide/tutorial-6-4-wheels-robot?tab-language=c++)

1. 设置robot node(根节点)
2. 给robot node添physics node(有了质量)和boundingObject(有了碰撞边界)
3. 给robot node 的 children field添加shape node(机器人的身体，一个长方体)
4. 给robot node 的 children field添加4个HingeJoint node,连接4个轮子
   1. HingeJoint node的device field添加RotationMotor(轮子有了马达)
   2. HingeJoint node的endPoint node 添加solid node，来定义轮子的形状，位置，物理性质
   3. HingeJoint node的jointParameters 添加HingeJointParameters，其中axis定义旋转轴，anchor定义旋转轴的位置(要和轮子的位置一致) 
5. 给robot node 的 children field添加 2个DistanceSensor node
   1. 可以在DistanceSensor node的children中添加 shape node来给传感器设定一个外形
   2. 设置DistanceSensor node的translation field和rolation field来设置传感器的位置和朝向。
      - 可以用：`ctrl+F10`或`View/Optional Rendering/Show DistanceSensor Rays`来显示传感器的朝向
6. 在robot node的controller field中选择 controller

- 控制4轮机器人

  ```c++
  #include <webots/DistanceSensor.hpp>
  #include <webots/Motor.hpp>
  #include <webots/Robot.hpp>
  
  #define TIME_STEP 64
  using namespace webots;
  
  int main(int argc, char **argv) {
    Robot *robot = new Robot();
      
    // initialize Sensors
    DistanceSensor *ds[2];
    char dsNames[2][10] = {"ds_right", "ds_left"};
    for (int i = 0; i < 2; i++) {
      ds[i] = robot->getDistanceSensor(dsNames[i]);
      ds[i]->enable(TIME_STEP);
    }
    // initialize motors
    Motor *wheels[4];
    char wheels_names[4][8] = {"wheel1", "wheel2", "wheel3", "wheel4"};
    for (int i = 0; i < 4; i++) {
      wheels[i] = robot->getMotor(wheels_names[i]);
      // 一个motor可以通过设置位置、速度、加速度或者力来驱动。
      // 这里通过设置速度来控制motor，所以位置的值设为INFINITY(即最终位置是无限远处，不构成控制)
      wheels[i]->setPosition(INFINITY);
      wheels[i]->setVelocity(0.0);
    }
    int avoidObstacleCounter = 0;
      
    while (robot->step(TIME_STEP) != -1) {
      double leftSpeed = 1.0;
      double rightSpeed = 1.0;
      if (avoidObstacleCounter > 0) {
        avoidObstacleCounter--;
        leftSpeed = 1.0;
        rightSpeed = -1.0;
      } else { // read sensors
        for (int i = 0; i < 2; i++) {
          // 这里sensor的返回值可以参考相关sensor node的文档
          // 还可以通过双击robot body来打开robot-window来实时查看传感器的返回值
          if (ds[i]->getValue() < 950.0)
            avoidObstacleCounter = 100;
        }
      }
      wheels[0]->setVelocity(leftSpeed);
      wheels[1]->setVelocity(rightSpeed);
      wheels[2]->setVelocity(leftSpeed);
      wheels[3]->setVelocity(rightSpeed);
    }
    delete robot;
    return 0;  // EXIT_SUCCESS
  }
  ```

- 将我们的robot做成一个proto原型

  这样就可以很方便在不同的world中使用这个robot。

  1. 在工程的protos folder中建一个xx.proto文档

  2. 任何proto文档都有如下格式：

     ```
     PROTO protoName [
       protoFields
     ]
     {
       protoBody
     }
     ```

     - `protoName`:是这个proto的名字
     - `protoFields`：定义了这个proto中可modifiable修改的fields
     - `protoBody`:是这个root node的定义

  3. 将xx.wbt文件中关于机器人的描述`Robot{xxx}`完全的复制到protoBody

     1. 注意这时候的translation,rotation,bodymass都是原型的初值，这导致新加入的node无法移动

     2. 为此将Robot{xx}中的translation,rotation,bodymass改成：

        ```
            translation IS translation
            rotation IS rotation
            mass IS bodyMass
        ```

  4. 此时添加节点就可以在`PROTO nodes (Current Project) / FourWheelsRobot (Robot)`中看到

  5. 设定字段

     ```
     PROTO FourWheelsRobot [
       field SFVec3f    translation  0 0 0
       field SFRotation rotation     0 0 1 0
       field SFFloat    bodyMass     1
     ]
     ```

     
