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
   
4.  使用 `ros2 node <command> -h` 可以获得更多使用细节

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
  
- 使用 `ros2 pkg <command> -h` 可以获得更多使用细节

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
  
- 文档

  https://colcon.readthedocs.io/en/released/index.html

## 手撸一个节点python

节点需要存在于功能包当中、功能包需要存在于工作空间当中。所以我们要想创建节点，就要先创建一个工作空间，再创建功能包。

1. 创建工作空间

   工作空间就是文件夹
   
   ```
   mkdir -p town_ws/src
   cd town_ws/src
   ```
   
2. 创建一个功能包

   创建一个叫village的功能包

   ```shell
   ros2 pkg create village --build-type ament_python --dependencies rclpy
   ```

   - pkg create创建包
   - --build-type 用来指定该包的编译类型，一共有3个可选项。
     - `ament_python`
     - `ament_cmake`默认项
     - `cmake`
   - --dependencies这个功能包的依赖，这里给了一个ROS2的python客户端接口rclpy

3. 创建节点文件

   用tree查看目录结构后，在`__init__.py`同级别目录下创建一个`point1.py`的文件

2种编程方式写节点：oop面向对象编程

### 1. 使用非oop方法编写一个节点

1. 一般步骤

   1. 导入库文件
   2. 初始化客户端库
   3. 新建节点
   4. spin循环节点
   5. 关闭

2. 编写point1.py

   ```python
   import rclpy
   from rclpy.node import Node
   
   def main(args=None):
       rclpy.init(args=args)   #初始化rclpy
       node = Node("point1")	#新建一个节点
       node.get_logger().info("hello I'm Yang")	
       rclpy.spin(node)		#保持节点运行。检测是否受到退出指令ctrl+C
       rclpy.schutdown()		#关闭rclpy
   ```

3. 修改src目录下的setup.py

   在entry_points中加入：

   ```python
       entry_points={
           'console_scripts': [
               "point1_node = village.point1:main"
           ],
       },
   )
   ```

4. 编译

   1. 在town_ws目录下打开终端编译节点`colcon build`
   2. source环境：`source install/setup.bash`
   3. 运行节点`ros2 run village point1_node`
   4. 在另一个终端下用`ros2 node list`查看现有节点

​    

### 2.使用oop方法编写一个节点

   1. 在上节基础上只需修改Point1.py

      ```python
      import rclpy
      from rclpy.node import Node
      
      
      class WriterNode(Node):
          def __init__(self,name):
              super().__init__(name)
              self.get_logger().info("Hi,I'm Yang")
      
      def main(args=None):
          rclpy.init(args=args)   
          node = WriterNode("node")
          rclpy.spin(node)
          rclpy.schutdown()
      ```

## 手撸一个节点c++

1. 创建工作空间

   `mkdir -p town/src`

2. 在工作空间src内创建功能包

   `ros2 pkg create village --build-type ament_cmake --dependencies rclcpp`

3. 在village/src下创建point.cpp，然后编写节点代码

### 1.使用pop面向过程方法

1. 编写point.cpp

   ```c++
   #include "rclcpp/rclcpp.hpp"
   
   int main(int argc, char **argv){
       rclcpp::init(argc,argv);
       auto node = std::make_shared<rclcpp::Node>("new"); //Create a node named point
       RCLCPP_INFO(node->get_logger(),"HI,I'm Yang");
       rclcpp::spin(node);
       rclcpp::shutdown();
       return 0;
   }
   ```

   - rclcpp/rclcpp.hpp在opt/ros/galactic下，vscode要识别需要加入地址Path中去

2. 修改CMakeLists.txt

   在最后加入

   ```
   /*让编译器编译point.cpp这个文件*/
   add_executable(new_node src/point.cpp)
   ament_target_dependencies(new_node rclcpp)
   
   /*要手动将编译好的文件安装到install/village_wang/lib/village_wang下*/
   install(TARGETS
           new_node
           DESTINATION lib/${PROJECT_NAME}
   )
   ```

3. 编译

   - 在town中编译节点`colcon build`
   - source环境`source install/setup.bash`

   - 运行节点`ros2 run village new_node`
   - `ros2 node list`查看现有节点

### 2.使用oop面向对象方式

只需修改point.cpp

```c++
#include "rclcpp/rclcpp.hpp"

class SingleDogNode:public rclcpp::Node
{
public:
	//构造函数
    SingleDogNode(std::string name) : Node(name){
        RCLCPP_INFO(this->get_logger(),"Hi,I'm Yang Xu");
    }
};



int main(int argc, char **argv){
    rclcpp::init(argc,argv);
    auto node = std::make_shared<SingleDogNode>("new"); //Create a node named point
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
```



