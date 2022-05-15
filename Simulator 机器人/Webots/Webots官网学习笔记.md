# 一、基本认识

- 一些概念
  - **World**: A world is a 3D description of the properties of robots and of their environment. 不包含controller的代码但会specify机器人所需要的controller的名字。Worlds保存在".wbt"文件内，而".wbt"文件在每一个Webots工程的"worlds"子目录下。
  - **Controller**: A controller is a computer program that controls a robot specified in a world file.可以用c,c++,java,python或者matlab来写。
    - 一个controller可以用于多个机器人，但一个机器人一次只能运行一个controller
  - **Supervisor Controller**: 普通controller的supervisor field字段设置为true。它通常用于执行只能由人类操作员而非机器人执行的操作。In contrast with a regular Robot controller, the Supervisor controller will have access to privileged有特权的 operations. 

- 一些操作

  - shift+鼠标左键：水平移动物体

  - alt+鼠标左键：给物体施加力。注意如果物体（node）质量为0，就无法施加力，所以要先设定质量。