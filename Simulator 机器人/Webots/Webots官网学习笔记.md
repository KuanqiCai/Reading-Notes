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

## Node

Nodes用的是VRML97 syntax语法，一系列的node共同构建了world。在Scene Tree栏中，可以看到现在world中所有的nodes。每一个node下都有各种field字段，用于设置该node(物体)的速度、大小、位置、controller等等一切属性。

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