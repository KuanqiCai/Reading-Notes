# ROS2 基本概念
## ROS2节点

1. 节点是什么

   节点是一个可执行文件，每一个节点只负责一个单独的模块化的功能（比如一个节点负责控制车轮转动，一个节点负责从激光雷达获取数据、一个节点负责处理激光雷达的数据、一个节点负责定位等等）

2. 节点之间如何交互

   ros2共有4种通信方式：

   1. 话题-topics

   2. 服务-services

   3. 动作-Action

   4. 参数-parameters

      ![](http://d2lros2foxy.fishros.com/chapt3/3.1ROS2%E8%8A%82%E7%82%B9%E4%BB%8B%E7%BB%8D/imgs/Nodes-TopicandService.gif)

3. 启动一个节点

   - `ros2 run <package_name> <executable_name>`

     启动 某一包 中的节点

   - 例：启动小乌龟模拟器

     `ros2 run turtlesim turtlesim_node`

   - 查看节点列表

     `ros2 node list`

   - 查看节点信息

     `ros2 node info <node_name>`

   - 重映射节点名称

     `ros2 run turtlesim turtlesim_node --ros-args --remap __node:=my_turtle`

## ROS2 工作空间和功能包

一个工作空间有多个功能包，一个功能包可以有多个节点。

- 工作空间：工作空间是包含若干个功能包的文件夹：

  创建一个工作空间，必须有src文件夹

  ```shell
  mkdir -p turtle_ws/src
  cd turtle_ws/src
  ```

- 功能包：

  - 存放节点的地方，根据编译方式不同，有3种类型

    1. ament_python，适用于python
    2. cmake,适用于c++
    3. ament_cmake,适用于c++，cmake的增强版

  - 与功能包相关的指令:ros2 pkg 

    ```
    create       Create a new ROS2 package
    executables  Output a list of package specific executables
    list         Output a list of available packages
    prefix       Output the prefix path of a package
    xml          Output the XML of the package manifest or a specific tag
    ```

    1. 创建功能包

       `ros2 pkg create <package-name>  --build-type  {cmake,ament_cmake,ament_python}  --dependencies <依赖名字>`

    2. 列出可执行文件

       列出所有

       `ros2 pkg executables`

       列出某个功能包的

       `ros2 pkg executables turtlesim`

    3. 列出所有的包

       `ros2 pkg list`

    4. 输出某个包所在路径的前缀,如小乌龟

       `ros2 pkg prefix turtlesim`

    5. 列出包的清单描述文件

       每一个功能包都有一个标配的manifest.xml文件，用于记录这个包的名字，构建工具，编译信息，拥有者，干啥用的等信息。通过这个信息，就可以自动为该功能包安装依赖，构建时确定编译顺序等

       `ros2 pkg xml turtlesim `

## ROS2 编译器 Colcon

- 安装

  `sudo apt-get install python3-colcon-common-extensions`

- 相关指令

  - 只编译一个包

    `colcon test --packages-select YOUR_PKG_NAME `

  - 不编译测试单元

    `colcon test --packages-select YOUR_PKG_NAME  --cmake-args -DBUILD_TESTING=0`

  - 运行编译的包的测试

    `colcon test`

  - 允许通过更改src下的部分文件来改变install

    每次调整python脚本时都不必重新build了

    `colcon build --symlink-install`

    

    

    