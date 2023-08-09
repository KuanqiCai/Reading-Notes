# Vrep 函数用法
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



# ROS2 通信机制

## 话题python

### 工具

- rqt_graph：

  ```
  //展示：3个终端分别输入
  ros2 run demo_nodes_py listener
  ros2 run demo_nodes_cpp talker
  rqt_graph
  ```

- CLI命令行界面工具

  `ros2 topic -h`查看所有的topic指令


### 话题的编程实现

- 编写一个话题发布者的流程
  1. 导入消息类型
  2. 声明并创建发布者
  3. 编写发布逻辑发布数据

- 在Python的OOP基础上将代码改为：

  ```python
  import rclpy
  from rclpy import timer
  from rclpy.node import Node
  
  # 1.Insert the type of info
  from std_msgs.msg import String
  
  class WriterNode(Node):
      def __init__(self,name):
          super().__init__(name)
          self.get_logger().info("Hi,I'm Yang")
  
          # 2.create and inititialize the members' properties 
          # create_publisher()创建发布者，参数1：方法类型，2：话题名称，3：消息队列长度
          # 这里的string类是从std_msgs.msg的。std_msgs是ros2自带的接口类型。可用ros2 interface package std_msgs查看所有消息类型。
          self.pub_novel = self.create_publisher(String,"The Base",10)
          
          # 3.set the release logic
          self.i = 0  # i is a counter
          timer_period = 5 # write a chapter every 5s
          self.timer = self.create_timer(timer_period,self.timer_callback)    #call the function timer_callback every 1 s
  
      def timer_callback(self):
          msg = String()
          msg.data = "The %d chapter: The %d time I save the world"%(self.i,self.i)
          self.pub_novel.publish(msg)     # release the novel 
          self.get_logger().info('LI4: I release "The Base":"%s"'%msg.data)
          self.i += 1
  
  def main(args=None):
      rclpy.init(args=args)   
      node = WriterNode("node")
      rclpy.spin(node)
      rclpy.schutdown()
  ```
  
- 单独编译village_li

  `colcon build --packages-select  village_li`

- 运行节点

  ```
  source install/setup.bash
  ros2 run village_li li4_node
  ```

### 话题订阅

- 创建话题订阅的一般流程

  1. 导入订阅的话题接口类型
  2. 创建订阅回调函数
  3. 声明并创建订阅者
  4. 编写订阅回调处理逻辑

- 在上诉代码基础上添加了创建订阅器的函数：

  `self.create_subscription(UInt32,"sexy_girl_money",self.recv_money_callback,10)`

  这句话的意思是创建订阅者，订阅话题`sexy_girl_money`,话题类型为`UInt32`,每次收到钱就去调用`self.recv_money_callback`函数存起来。

- 完整代码：

  ```python
  #!/usr/bin/env python3
  import rclpy
  from rclpy.node import Node
  # 导入话题消息类型
  from std_msgs.msg import String,UInt32
  
  class WriterNode(Node):
      """
      创建一个李四节点，并在初始化时输出一个话
      """
      def __init__(self,name):
          super().__init__(name)
          self.get_logger().info("大家好，我是%s,我是一名作家！" % name)
          # 创建并初始化发布者成员属性pubnovel
          self.pubnovel = self.create_publisher(String,"sexy_girl", 10) 
  
  
          # 创建定时器成员属性timer
          self.i = 0 # i 是个计数器，用来算章节编号的
          timer_period = 5  #每5s写一章节话
          self.timer = self.create_timer(timer_period, self.timer_callback)  #启动一个定时装置，每 1 s,调用一次time_callback函数
  
  
          # 账户钱的数量
          self.account = 80
          # 创建并初始化订阅者成员属性submoney
          self.submoney = self.create_subscription(UInt32,"sexy_girl_money",self.recv_money_callback,10)
          
  
      def timer_callback(self):
          """
          定时器回调函数
          """
          msg = String()
          msg.data = '第%d回：潋滟湖 %d 次偶遇胡艳娘' % (self.i,self.i)
          self.pubnovel.publish(msg)  #将小说内容发布出去
          self.get_logger().info('李四:我发布了艳娘传奇："%s"' % msg.data)    #打印一下发布的数据，供我们看
          self.i += 1 #章节编号+1
  
  
      def recv_money_callback(self,money):
          """
          4. 编写订阅回调处理逻辑
          """
          self.account += money.data
          self.get_logger().info('李四：我已经收到了%d的稿费' % self.account)
  
  
  def main(args=None):
      """
      ros2运行该节点的入口函数，可配置函数名称
      """
      rclpy.init(args=args) # 初始化rclpy
      node = WriterNode("li4")  # 新建一个节点
      rclpy.spin(node) # 保持节点运行，检测是否收到退出指令（Ctrl+C）
      rclpy.shutdown() # rcl关闭
  
  ```

- 编译

  ```
  colcon build --packages-select  village_li
  source install/setup.bash
  ros2 run village_li li4_node
  ```

- 然后一个终端发布话题，一个终端订阅：

  发布：

  `ros2 run village_li li4_node`

  订阅：

  `ros2 topic pub  /sexy_girl_money std_msgs/msg/UInt32 "{data:10}" `

## 话题c++

- 在节点中SingleDogNode类继承了Node类

  `class SingleDogNode : public rclcpp::Node`

- 这次用到的3种继承来的方法：

  1. 创建一个话题订阅者，用于拿到艳娘传奇的数据

     ```c++
     rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_;
     sub_ = this->create_subscription<std_msgs::msg::String>("sexy_girl", 10, std::bind(&SingleDogNode::topic_callback, this, _1));
     ```

  2. 创建一个话题发布者的能力，用于给李四发送稿费

     ```c++
     rclcpp::Publisher<std_msgs::msg::UInt32>::SharedPtr pub_;
     pub_ = this->create_publisher<std_msgs::msg::UInt32>("sexy_girl_money",10);
     ```

  3. 获取日志打印器的能力

     ```c++
     // 打印一句自我介绍
     RCLCPP_INFO(this->get_logger(), "大家好，我是单身汉王二.");
     ```

  4. 更多的方法：https://docs.ros2.org/foxy/api/rclcpp/index.html

### 代码实现

- 代码实现一般流程

  1. 导入订阅的话题接口类型
  2. 创建订阅回调函数
  3. 声明并创建订阅者
  4. 编写订阅回调处理逻辑

- 在town工作空间打开village中的point.cpp

  ```c++
  #include "rclcpp/rclcpp.hpp"
  #include "std_msgs/msg/string.hpp"
  #include "std_msgs/msg/u_int32.hpp"
  
  using std::placeholders::_1;
  using std::placeholders::_2;
  
  /*
      创建一个类节点，名字叫做SingleDogNode,继承自Node.
  */
  class SingleDogNode : public rclcpp::Node
  {
  
  public:
      // 构造函数,有一个参数为节点名称
      SingleDogNode(std::string name) : Node(name)
      {
          // 打印一句自我介绍
          RCLCPP_INFO(this->get_logger(), "大家好，我是单身狗%s.", name.c_str());
           // 创建一个订阅者来订阅李四写的小说，通过名字sexy_girl
          sub_novel = this->create_subscription<std_msgs::msg::String>("sexy_girl", 10, std::bind(&SingleDogNode::topic_callback, this, _1));
      }
  
  private:
      // 声明一个订阅者（成员变量）,用于订阅小说
      rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_novel;
  
      // 收到话题数据的回调函数
      void topic_callback(const std_msgs::msg::String::SharedPtr msg)
      {
          RCLCPP_INFO(this->get_logger(), "朕已阅：'%s'", msg->data.c_str());
      };
  
  };
  
  int main(int argc, char **argv)
  {
      rclcpp::init(argc, argv);
      /*产生一个Wang2的节点*/
      auto node = std::make_shared<SingleDogNode>();
      /* 运行节点，并检测退出信号*/
      rclcpp::spin(node);
      rclcpp::shutdown();
      return 0;
  }
  ```

- 编译运行

  `colcon build --packages-select village`

  source 运行

  ```
  source install/setup.bash 
  ros2 run  village new_node 
  ```

- 在新终端运行li4节点

  ```
  source install/setup.bash 
  ros2 run  village_li  li4_node 
  ```

### 增加发布稿费功能

- 给wang2定义一个发布者，用于给li4发稿费

  `pub_ = this->create_publisher<std_msgs::msg::UInt32>("sexy_girl_money",10);`

  这里提供了三个参数，分别是该发布者要发布的话题名称（`sexy_girl_money`）、发布者要发布的话题类型（`std_msgs::msg::UInt32`）、Qos（10）

- 完整代码

  ```c++
  #include "rclcpp/rclcpp.hpp"
  #include "std_msgs/msg/string.hpp"
  #include "std_msgs/msg/u_int32.hpp"
  
  
  using std::placeholders::_1;
  using std::placeholders::_2;
  
  /*
      创建一个类节点，名字叫做SingleDogNode,继承自Node.
  */
  class SingleDogNode : public rclcpp::Node
  {
  
  public:
      // 构造函数,有一个参数为节点名称
      SingleDogNode(std::string name) : Node(name)
      {
          // 打印一句自我介绍
          RCLCPP_INFO(this->get_logger(), "大家好，我是单身狗%s.", name.c_str());
           // 创建一个订阅者来订阅李四写的小说，通过名字sexy_girl
          sub_novel = this->create_subscription<std_msgs::msg::String>("sexy_girl", 10, std::bind(&SingleDogNode::topic_callback, this, _1));
          // 创建发布者
          pub_money = this->create_publisher<std_msgs::msg::UInt32>("sexy_girl_money", 10);
      }
  
  private:
      // 声明一个订阅者（成员变量）,用于订阅小说
      rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_novel;
  
      // 声明一个发布者（成员变量）,用于给李四钱
      rclcpp::Publisher<std_msgs::msg::UInt32>::SharedPtr pub_money;
  
      // 收到话题数据的回调函数
      void topic_callback(const std_msgs::msg::String::SharedPtr msg)
      {
          // 新建一张人民币
          std_msgs::msg::UInt32 money;
          money.data = 10;
          // 发送人民币给李四
          pub_money->publish(money);
  
          RCLCPP_INFO(this->get_logger(), "朕已阅：'%s'，打赏李四：%d 元的稿费", msg->data.c_str(), money.data);
      };
  
  };
  
  ```

  

