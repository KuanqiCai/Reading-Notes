# ROS基础

## *catkin的一些操作

- 编译的抽象程度,高到低
  - catkin: workspace(Package)会在工作空间中单独的构建每个包
  - cmake: CMakeLists.txt自己写编CMakeLists.txt来自动生成makefile
  - make: makefile通过makefile来批量的执行g++编译命令
  - gcc/g++: .cpp->.o->可执行文件

```shell
catkin build -DPYTHON_EXECUTABLE=/usr/bin/python3
#会将工作空间里所有的包同时单独（isolated）编译，编译过程互不影响
#首次build要加后面一行

catkin init
#可以初始化workspace，而且初始化后，在workspace下的任何一个子目录里使用catkin工具，都相当于在workspace文件夹下使用，它会自动感知workspace。

catkin config
#可以查看workspace的结构和参数。

catkin config --merge-devel
#可以使得每个包编译得到的devel产物都在同一个devel文件夹里，当然也可以用

catkin config --isolate-devel
#使得各个包的devel分开。

catkin list
#可以查看工作空间中有哪些ros包。

catkin clean
#相当于rm -r ${build} ${devel}，但是避免了rm -r这种危险的操作！
```

## *从0搭建一个project

1. 创建工作空间:`mkdir -p ~/test_ws/src`

2. 创建包：`catkin_create_pkg test_ros_pkg std_msgs roscpp`

3. 写节点

   1. 包目录下的src目录下编写c++

   2. 包目录下的CMakeLists编写,比如添加

      ```
      add_executable(superROS src/supeROS.cpp)
      target_link_libraries(superROS ${catkin_LIBRARIES})
      ```

4. 编译：`catkin build`

   1. 首次编译后，记得在 ~/.bashrc文件尾部加入`source ~/Destktop/as_ws/devel/setup.bash`

      这样就不用每打开一个bash都要输一遍`source devel/setup.bash`

5. 重复3，4

运行:

1. 启动master:`roscore`
2. 跑节点：具体看节点那一节
3. 分析数据:
   1. rostopic list, rostopic info, rostopic echo [topic]
   2. rosnode list, rosnode info [node]
   3. rosmsg list,
   4. rospkg list,

## 一、创建工作空间Workspace

```shell
## 创建工作空间
$ mkdir -p ~/catkin_ws/src
$ cd ~/catkin_ws/
$ catkin_make
## 对于ros时Melodic版本的Python 3用户
$ catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
## 激活工作空间。一个系统可以有多个工作空间，但只能有1个处于激活状态
$ source devel/setup.bash
## 检查环境变量查看当前激活的是哪个工作空间
$ echo $ROS_PACKAGE_PATH
```

- src源文件空间：放置各个功能包和一个关于这些功能包的配置文件CMakeLists.txt用于编译的配置。

- build编译空间：放置CMake和catkin编译功能包时产生的缓存、配置、中间文件等

- devel开发空间：放置编译好的可执行程序。

- 对于激活工作空间：

  ```
  ## 每一次打开终端都要输入一遍
  $ source devel/setup.bash
  
  ## 如果之后都只开发该工作空间，可以直接将它写入.bashrc文件
  $ gedit ~/.bashrc
  # 然后在该文件中末尾加入：
  source ~/Destktop/as_ws/devel/setup.bash
  ```
## 二、创建包Package

- 一个catkin package必须具备：

  1. 包含package.xml文件
  2. 包含CMakeLists.txt文件
  3. 每一个包必须有自己的文件夹

  ```
  my_package/
    CMakeLists.txt
    package.xml
  ```

- **在catkin Workspace中可以创建多个包**。

  ```
  workspace_folder/        -- WORKSPACE
    src/                   -- SOURCE SPACE
      CMakeLists.txt       -- 'Toplevel' CMake file, provided by catkin
      package_1/
        CMakeLists.txt     -- CMakeLists.txt file for package_1
        package.xml        -- Package manifest for package_1
      ...
      package_n/
        CMakeLists.txt     -- CMakeLists.txt file for package_n
        package.xml        -- Package manifest for package_n
  ```

  - 用catkin_create_pkg脚本创建新的catkin包

    用法：`$ catkin_create_pkg <package_name> [depend1] [depend2] [depend3]`

    例子：`$ catkin_create_pkg beginner_tutorials std_msgs rospy roscpp`在工作空间catkin_ws/src中创建包

- **编译catkin workspace并sorcing setup文件**

  因为roscd只能在已经被加入ros环境的地址中寻找

  ```shell
  $ cd ~/catkin_ws
  $ catkin_make
  ## 将工作空间加入ros环境
  $ source ./catkin_ws/devel/setup.bash
  ```

- **package dependencies**

  - 检查当前包直接依赖First-order dependencies

    `$ rospack depends1 beginner_tutorials `

    这些依赖被储存在package.xml中,用如下命令查看

    ```
    $ roscd beginner_tutorials
    $ cat package.xml
    ```

  - 检查间接依赖Indirect dependencies

    一个dependency通常有自己的dependecies

    `$ rospack depends1 rospy`

    直接显式包所有的依赖：直接+间接依赖

    `$ rospack depends beginner_tutorials`

### 定制自己的包package.xml

通过修改**package.xml**来customize包。

编译后生成的package.xml会在新包的文件中。比如~/Desktop/catkin_ws/src/beginner_tutoria

1. description tag:描述这个包干什么用的

   ```xml
   5   <description>The beginner_tutorials package</description>
   ```

2. maintainer tags:告诉别人谁拥有这个包，并如何联系

   ```xml
   10   <maintainer email="you@yourdomain.tld">Your Name</maintainer>
   ```

3. license tags:这个包用的许可协议

   ```xml
   16   <license>BSD</license>
   ```

4. dependencies tags:这个包用到的依赖

   ```xml
   51   <buildtool_depend>catkin</buildtool_depend>
   52 
   53   <build_depend>roscpp</build_depend>
   54   <build_depend>rospy</build_depend>
   55   <build_depend>std_msgs</build_depend>
   56 
   57   <exec_depend>roscpp</exec_depend>
   58   <exec_depend>rospy</exec_depend>
   59   <exec_depend>std_msgs</exec_depend>
   ```

## 三、构建包catkin_make

- 首先使用catkin_make 

  用法：`$ catkin_make [make_targets] [-DCMAKE_VARIABLES=...]`在catkin工作空间中用

  例子：`$ catkin_make`构建src下所有的包

  ​			`$ catkin_make -DCATKIN_WHITELIST_PACKAGES="package1;package2"`构建src下特定的包

- 编译后会有2个新的文件夹出现在workspace中

    1. **build**:the default location of the [build space](http://wiki.ros.org/catkin/workspaces#Build_Space) and is where `cmake` and `make` are called to configure and build your packages.
    2. **devel**:the default location of the [devel space](http://wiki.ros.org/catkin/workspaces#Development_.28Devel.29_Space), which is where your executables and libraries go before you install your packages.

- 如果想要构建的包在另一个src目录，比如my_src

    例子：`$ catkin_make --source my_src`

  ## $$ROS文件系统Filesystem Concepts

- **Packages**: Packages are the software organization unit of ROS code. Each package can contain libraries, executables可执行文件, scripts脚本, or other artifacts代码生成物.

- **Manifests**: A manifest is a description of a *package*. It serves to define dependencies between *packages* and to capture meta information about the *package* like version, maintainer, license, etc...

### Filesystem Tools
注意：所有的ros工具只会找到在[ROS_PACKAGE_PATH](http://wiki.ros.org/ROS/EnvironmentVariables#ROS_PACKAGE_PATH)中列出的ros包。可用`$ echo $ROS_PACKAGE_PATH`查看

- **rospack**

  用于找packages的信息。

  用法：`$ rospack find [package_name]`

  例子：`$ rospack find roscpp`

- **roscd**

  rosbash套件的一部分，用于前往packages的目录directory。

    用法：`$ roscd <package>[/subdir]`

    例子：`$ roscd roscpp`  前往roscpp包所在的目录，可用`pwd`查看当前工作目录。

    			`$ roscd roscpp/cmake  `前往roscpp包所在的目录中cmake子目录。

  - `roscd log`如果运行过ros程序后，可以查看日志文件

- **rosls**

  相当于ls，直接显式包里的内容

  	用法：`$ rosls <package-or-stack>[/subdir]`
  	
  	例子：`rosls roscpp_tutorials`

 ## $$节点node和节点管理器master

一个包里可以有多个可执行文件，可执行文件在运行之后就成了一个进程process，这个进程在ROS中就叫做节点。

通常一个node负责机器人的某一个单独的功能。

- 一些概念
  - [Nodes](http://wiki.ros.org/Nodes): A node is an executable process that uses ROS to communicate with other nodes.
  - [Messages](http://wiki.ros.org/Messages): ROS data type used when subscribing or publishing to a topic.
  - [Topics](http://wiki.ros.org/Topics): Nodes can *publish* messages to a topic as well as *subscribe* to a topic to receive messages.
  - [Master](http://wiki.ros.org/Master): Name service for ROS (i.e. helps nodes find each other)
  - [rosout](http://wiki.ros.org/rosout): ROS equivalent of stdout/stderr
  - [roscore](http://wiki.ros.org/roscore): Master(provides name service for ROS) + rosout + parameter server 
  
- `roscore`启动节点管理器**master**
  
  - 使用ROS第一步就是输入`roscore`
  - 用于节点管理，是ros节点运行的前提。
  - roscore命令会启动3个进程，可用`ps -ef|grep ros`查看进程
    - roscore(父进程启动后面2个)
    - rosmaster(节点管理器)
    - rosout(log输出管理)
  
- `rosnode`命令详细列表

  - `rosnode list`查看当前运行着的节点

    这时候返回：`/rosout`

  - `rosnode info /rosout`返回特定节点的信息
  
  - `rosnode kill /rosout` 结束某个node
  
  - `rosnode ping`测试连接节点
  
  - `rosnode machine`列出在特定机器或列表机器上运行的节点
  
  - `rosnode cleanup`清除不可到达节点的注册信息

### 简单的例子

#### 1.Publisher

```c++
#include "ros/ros.h"
#include "std_msgs/String.h"

#include <sstream>

//main function
int main(int argc, char **argv)
{

  //start node
  ros::init(argc, argv, "SupeROS");

  //create variable for node
  ros::NodeHandle n;

  //declare publishers (employees)
  ros::Publisher MrFish = n.advertise<std_msgs::String>("fish", 1000);
  ros::Publisher MrVeggies = n.advertise<std_msgs::String>("veggies", 1000);
  ros::Publisher MrFruits = n.advertise<std_msgs::String>("fruits", 1000);


  ros::Rate loop_rate(10);

  int count = 0;
  while (ros::ok())
  {
    // declare msg for publishing data
    std_msgs::String msg_fisher;
    std_msgs::String msg_veggies;
    std_msgs::String msg_fruits;

    //define possible options for food
    std::vector<std::string> fisher_options = { "tuna", "salmon", "shark" };
    std::vector<std::string> veggies_options = { "onion", "potatoes", "carrots" };
    std::vector<std::string> fruit_options = { "bananas", "apples", "grapes" };

    //store current option
    if (count == 3){count = 0;} //reset counter if longer than amount of options
    msg_fisher.data = fisher_options[count];
    msg_veggies.data = veggies_options[count];
    msg_fruits.data = fruit_options[count];


    //publish
    MrFish.publish(msg_fisher);
    MrVeggies.publish(msg_veggies);
    MrFruits.publish(msg_fruits);

    ros::spinOnce();

    loop_rate.sleep();

    ++count;
  }


  return 0;
}

```

#### 2.Client

```c++
#include "ros/ros.h"
#include "std_msgs/String.h"

void customerCallback1(const std_msgs::String& msg){
        if (msg.data == "carrots")
        {
            //print in command window
            ROS_INFO("Customer 1: I got carrots!");
        }
}
void customerCallback2(const std_msgs::String& msg){
        if (msg.data == "tuna")
        {
            //print in command window
            ROS_INFO("Customer 2: I got tuna!");
        }
}

int main(int argc, char **argv)
{

  ros::init(argc, argv, "Clients");

  ros::NodeHandle n;

  ros::Subscriber customer1_sub = n.subscribe("veggies", 1000, customerCallback1);
  ros::Subscriber customer2_sub = n.subscribe("fish", 1000, customerCallback2);

  ros::spin();

  return 0;
}

```

#### 3.将通信包装成一个类

- 把所有的变量都包装在一个类中
  - 一个实例的回调函数就可以调用同一个变量
  - Subscriber和Publisher也只在类实例化的时候生成一次

```c++
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/Float64.h"



//Class defining clients of supermarket
class SupeROS_clients{

  //Declare ROS things
  ros::NodeHandle nh_;
  ros::Subscriber customer_sub_;
  ros::Publisher customer_pub_;

  //Declare customer identifier
  float customer_identifier_;

  //Declare string variables to store desired products
  std::string desired_product_;

public:

  //Constructor
  SupeROS_clients(float customer_identifier, std::string desired_product, std::string desired_topic)
  {
    //Store class identifier
    customer_identifier_ = customer_identifier;

    //Store desired product
    desired_product_ = desired_product;

    //Define subsciber and publisher
    customer_pub_ = nh_.advertise<std_msgs::Float64>("/Money", 1);
    customer_sub_ = nh_.subscribe(desired_topic, 1000, &SupeROS_clients::customerCallback, this);
  }

  //Callback where we check if the employee offers what customers wants and pay back
  void customerCallback(const std_msgs::String& msg){
        if (msg.data == desired_product_)
        {
                //pay back
                std_msgs::Float64 coin_msg;
                coin_msg.data = 1.0;
                customer_pub_.publish(coin_msg);

                //print in command window
                ROS_INFO("Customer %f: I got %s and I paid back!", customer_identifier_, desired_product_.c_str());
        }
  }
};

//Main function
int main(int argc, char **argv)
{
        // initialize node
        ros::init(argc, argv, "Clients");
        ros::NodeHandle nh_private("~");
        
        //initialize classes
        SupeROS_clients Clients_node1(1.0, "tuna","fish"), Clients_node2(2.0, "carrots","veggies");
        
        ros::spin();
}

```

### 用`rosrun`启动单个节点

  - 命令：`rosrun [--prefix cmd] [--debug] package_name node_name [ARGS]`

    会自动寻找package下的名为EXECUTABLE的可执行程序，并将可选参数ARGS传入。

    - 例子：
    
      `$ rosrun turtlesim turtlesim_node`
      
      `$ rosrun turtlesim turtlesim_node __name:=my_turtle`给节点单独命名
      
    - 例子：在GDB(程序调试器)下运行ros程序
    
      `$ rosrun --prefix 'gdb -ex run --args' pkg_name node_name`

### 用roslaunch启动多个节点

用法：`$ roslaunch [package] [filename.launch]`

1. 回到工作空间`$ cd ~/catkin_ws`

2. 将当前环境导入ROS环境`source devel/setup.bash`

3. 寻找之前创建好的包`$ roscd ~/catkin_ws`

4. 创建一个launch文件夹`$ mkdir launch` 并`$ cd launch`

5. 创建一个启动文件turtlemimic.launch,输入如下代码

   ```xml
   切换行号显示
   	 <!--launch tag说明这个文件是一个launch file-->
      1 <launch>
      2 
          <!--这里开始2个命名空间为turtlesim1/2的group，都用名字为sim的turtlesim节点-->
      3   <group ns="turtlesim1">
      4     <node pkg="turtlesim" name="sim" type="turtlesim_node"/>
      5   </group>
      6 
      7   <group ns="turtlesim2">
      8     <node pkg="turtlesim" name="sim" type="turtlesim_node"/>
      9   </group>
     10   <!--这里开始一个叫mimic并有2个话题input/output的节点，这里会把节点顺序变为turtlesim1->mimic-> turtlesim2-->
     11   <node pkg="turtlesim" name="mimic" type="mimic">
     12     <remap from="input" to="turtlesim1/turtle1"/>
     13     <remap from="output" to="turtlesim2/turtle1"/>
     14   </node>
     15 
     16 </launch>
   ```

6. 运行这个launch file:`$ roslaunch beginner_tutorials turtlemimic.launch`

7. 在新终端给他们提供命令：`$ rostopic pub /turtlesim1/turtle1/cmd_vel geometry_msgs/Twist -r 1 -- '[2.0, 0.0, 0.0]' '[0.0, 0.0, -1.8]'`

8. 使用`$ rqt_graph`查看节点关系.

launch文件包含的标签

```xml
<launch>    	<!--根标签-->
<node>    		<!--需要启动的node及其参数-->
<include>    	<!--包含其他launch-->
<machine>    	<!--指定运行的机器-->
<env-loader>    <!--设置环境变量-->
<param>    		<!--定义参数到参数服务器-->
<rosparam>    	<!--启动yaml文件参数到参数服务器-->
<arg>    		<!--定义变量-->
<remap>    		<!--设定参数映射-->
<group>   		<!--设定命名空间-->
</launch>    	<!--根标签-->
```

### 转换roslaunch和rosrun

1. **例1**

​	roslaunch中：

```xml
<node pkg="tf" type="static_transform_publisher" name="av1broadcaster" args ="1 0 0 0 0 0 1 world av1 100" />
```

​	rosrun中：

```shell
rosrun tf static_transform_publisher 1 0 0 0 0 0 1 world av1 100 __name:=av1broadcaster
```

2. **例2**：roslaunch中可以设定某个节点的启用条件，rosrun因为是人为一条条输入就没必要

```xml
<launch>
	<arg name = "static" default = "false" />
    ...
    <!--只在unless条件满足时才执行，这里默认false不执行，需要在命令行中手动输入static:=true-->
	<!--roslaunch two_drones_pkg two_drones.launch static:=true-->
    <node pkg="two_drones_pkg" type="frames_publisher_node" name="frames_publisher_node" unless="$(arg static)"/>
    ...
</launch>
```

3. **例3**:

   roslaunch中运用find快速定位文件

   ```xml
   <node name="rviz" pkg="rviz" type="rviz" args="-d $(find two_drones_pkg)/config/default.rviz"/>
   ```

   - 正常的agrs="$(find ..).."这里是没有参数-d的

     `rosrun rviz rviz --help`解释：-d用于展示这个配置文件

   rosrun中:

   ```shell
   rosrun rviz rviz -d ~/Desktop/as_ws/src/two_drones_pkg/config/default.rviz		
   ```

   

## $$ROS通信框架

### 0)不同通信框架的对比

ROS的通信方式有以下四种：

- Topic 主题
- Service 服务
- Parameter Service 参数服务器
- Actionlib 动作库

#### Topic vs Services

|   名称   |              Topic               |              Service              |
| :------: | :------------------------------: | :-------------------------------: |
| 通信方式 |             异步通信             |             同步通信              |
| 实现原理 |              TCP/IP              |              TCP/IP               |
| 通信模型 |        Publish-Subscribe         |           Request-Reply           |
| 映射关系 |    Publish-Subscribe(多对多)     |      Request-Reply（多对一）      |
|   特点   | 接受者收到数据会回调（Callback） | 远程过程调用（RPC）服务器端的服务 |
| 应用场景 |       连续、高频的数据发布       |     偶尔使用的功能/具体的任务     |
|   举例   |     激光雷达、里程计发布数据     |    开关传感器、拍照、逆解计算     |

### 1)话题Topic

对于实时性、周期性的消息，使用topic来传输是最佳的选择。topic是一种点对点的单向通信方式，这里的“点”指的是node。

topic要经历下面几步的初始化过程：

1. publisher节点和subscriber节点都要到节点管理器master进行注册
2. publisher会发布topic，subscriber在master的指挥下会订阅该topic

回调callback：Subscriber接收消息会进行处理。即提前定义好了一个处理函数(代码中)

如图所示整个过程都是单向的：node1、node2两者都是各司其责，不存在协同工作，我们称这样的通信方式是**异步**的。

topic可以同时有多个subscribers，也可以有多个publishers。如/rosout,/tf等。

​	![](https://github.com/Fernweh-yang/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/topic-stru.jpg?raw=true)

- 小案例

  - 输入`$ roscore`
  - 新终端运行节点`$ rosrun turtlesim turtlesim_node`
  - 新终端运行节点`$ rosrun turtlesim turtle_teleop_key`

- 话题topics的相关命令：

  上述案例中2个节点`turtlesim_node`和`turtle_teleop_key`就是通过话题topic来实施通信的：

  1. 发布者Publisher:`turtlesim_node`  **publishing** key strokes键给topic。

  2. 接收者Subscriber:`turtle_teleop_key` 从topic那里**subscribes** 这个keystrokes。

  - `rqt_graph`

    用于展示各个节点之间的关系

    新终端输入`$ rosrun rqt_graph rqt_graph`可看到两个节点通过话题`/turtle1/cmd_vel`联系在一起

  - `rostopic`

    用于获取ros话题的消息。可用`$ rostopic -h`查看帮助文档

    - `rostopic echo topic_name`：显示某个topic的内容

      1.新终端输入`rostopic echo /turtle1/cmd_vel` 

      2.再返回`turtle_teleop_key`的终端进行控制。`rostopic`终端就会出现传输给话题的消息。

    - `rostopic list `列出当前所有的topic

    - `rostopic info topic_name`显示某个topic的属性信息
    
    - `rostopic pub topic_name ` 向某个topic发布内容
    
    - `rostopic bw topic_name`查看某个topic的贷款
    
    - `rostopic hz topic_name`查看某个topic的频率
    
    - `rostopic find topic_type` 查看某个类型的topic
    
    - `rostopic type topic_name`查看某个topic的类型`msg` 

- ROS Messages消息，即下面的msg

  各节点通信发送的都是相同类型的message。一个话题的类型是由发布者Publisher的消息message类型决定的

  - `rostopic type [topic]`

    用于返回某一个话题的数据类型

    1. `rostopic type /turtle1/cmd_vel`得到话题的数据类型为：`geometry_msgs/Twist`
    2. `rosmsg show geometry_msgs/Twist` 显式这个数据类型的具体细节
    3. 也可以直接显式具体细节`$ rostopic type /turtle1/cmd_vel | rosmsg show`

  - `rostopic pub [topic] [msg_type] [args]` 

    用于将数据发送到一个当前推荐的话题上去。

    例：`$ rostopic pub -1 /turtle1/cmd_vel geometry_msgs/Twist -- '[2.0, 0.0, 0.0]' '[0.0, 0.0, 1.8]'`告诉乌龟以线速度2.0,角速度1.8运动。

    - `rostopic pub`将消息发布给指定的topic
    - `-1`只发送一条消息后就退出
    - `/turtle1/cmd_vel`topic名字
    - `geometry_msgs/Twist`发送的消息数据类型，由2个数组组成，每个数组3个浮点数。
    - `--`说明后面的arguments参数都不是option。

    

  - `rostopic hz [topic]`

    用于查看节点publish的速率rate。例：`rostopic hz /turtle1/pose`

  - `rqt_plot`

    用于绘制发布在话题上的scrolling time plot滚动时间图。例`rosrun rqt_plot rqt_plot`

### **ROS msg和srv

#### 1.msg,srv简介

- **msg**: 

  topics有严格的格式要求，这种数据格式就是message(msg)。所以msg即topic内容的数据类型，也称之为topic的格式标准。

  msg files是描述the fields of a ROS message的简单文本文件。他们以不同的语言来生成messages的源代码。储存在msg directory目录。

  - msg由一行行field type字段类型和field name字段名称组成，可用的字段类型:

    ```
    -int8, int16, int32, int64 (plus uint*)
    -float32, float64
    -string
    -time, duration
    -other msg files
    -variable-length array[] and fixed-length array[C]
    ```

  - msg还有一个特殊的类型header:包含一个timestamp时间戳和coordinate frame information坐标系信息。

  - 一个包含一个Header,一个string primitive,2个 other msg 文件

    ```
    string primitive  Header header
    string child_frame_id
    geometry_msgs/PoseWithCovariance pose
    geometry_msgs/TwistWithCovariance twist
    ```

    

- **srv**:

  区别于topic传输的数据格式由msg描述，service的数据格式由srv文件来描述。

  srv files描述了一个服务，包含2个部分：request 和 response。储存在srv directory。
  
  - src也由一行行field type 和 field name组成。request和response由"---"隔开。
  
    ```
    int64 A
    int64 B
    ---
    int64 Sum
    ```
  
    A,B是request,Sum是response

#### 2.使用msg

- 一些[常见msg](https://sychaichangkun.gitbooks.io/ros-tutorial-icourse163/content/chapter3/3.5.html)

  [ROS Message Types](http://wiki.ros.org/geometry_msgs)

- 相关命令：

  - `rosmsg list`: 列出系统上所有的msg

  - `rosmsg show msg_name`： 显示某个msg的内容

    - 例子：`$ rosmsg show beginner_tutorials/Num `

      如果忘了包名：`$ rosmsg show Num`

- 创建我们自己的msg
  
  **1.创建一个msg文件**
	
	```
	$ roscd beginner_tutorials
	$ mkdir msg
	$ echo "int64 num" > msg/Num.msg	#输出消息到一个新的文件
  ```
  
   **2.还可以`$ rosed beginner_tutorials Num.msg`,然后在里面输入**
  
    ```
    string first_name
    string last_name
    uint8 age
    uint32 score
    ```
  
  **3.编写package.xml和CMakeList**
  
  **3.1**：package.xml加入
  
  ```xml
  <!--在build time我们需要message_generation-->
  <build_depend>message_generation</build_depend>
  <!--在run time我们需要message_runtime-->
  <exec_depend>message_runtime</exec_depend>
  ```
  
  **3.2**：CMakeLists.txt
  
  ```cmake
  # 查找依赖的包
  find_package(catkin REQUIRED COMPONENTS
     roscpp	//必备
     rospy	//必备
     message_generation	//必备
     std_msgs	//我们自己写的msg中用到的其他msg
  )
  
  # 指定我们的msg文件
  add_message_files(
    FILES
    Num.msg
  )
  
  # 指定我们的msg用到的依赖项
  generate_messages(
    DEPENDENCIES
    std_msgs
  )
  
  # 设置运行依赖
  catkin_package(
   ...
    CATKIN_DEPENDS message_runtime ...
   ...)
  ```
  
- 在a_package中引用b_package里自定义的my_message.msg

  1. 首先在a_package的CMakeLists.txt文件中的find_package部分添加b_package:

     ```cmake
     find_package(catkin REQUIRED COMPONENTS
       roscpp
       rospy
       std_msgs
       b_package  #添加这一条
     )
     ```

  2. 然后在package.xml文件中添加以下代码,注意需要根据实际情况按相应的格式配置:

     ```xml
     <build_depend>b_package</build_depend>
     <build_export_depend>b_package</build_export_depend>
     <exec_depend>b_package</exec_depend>
     ```

  3. c++代码中

     ```c++
     #include <ros/ros.h>
     // b_packages::Test消息类型
     #include <b_packages/Test.h>
     ```

#### 3.使用srv

一些[常见srv](https://sychaichangkun.gitbooks.io/ros-tutorial-icourse163/content/chapter4/4.5.html)

- 相关的命令：

  - `$ rossrv show <service type>`：显示某一个服务描述

    - 例：`$ rossrv show beginner_tutorials/AddTwoInts````
  - `rossrv list`：列出所有的服务
  - `rossrv md5`：显示服务md5sum
  - `rossrv package`：列出包中的服务
  - `rossrv packages` ：列出包含服务的包

- 创建一个srv文件

  **1. 创建srv文件夹**

  ```
  $ roscd beginner_tutorials
  $ mkdir srv
  ```

  **2. 用roscp从另一个包那复制一个srv文件来**

  使用：`$ roscp [package_name] [file_to_copy_path] [copy_path]`

  例子：从rospy_tutorials包那复制它的srv到当前文件夹

  ```
  $ roscp rospy_tutorials AddTwoInts.srv srv/AddTwoInts.srv
  ```

  **3. 确保srv文件能转为c++,python或其他语言**

  ​	**3.1**: 编辑包内的package.xml,加入如下两行（当然在创建msg时已设置）

  ```xml
  <build_depend>message_generation</build_depend>
  <exec_depend>message_runtime</exec_depend>
  ```
  
  ​	**3.2**: 编辑包内的CMakeLists.txt（当然在创建msg时已设置）
  ​		(1): 将message_generation依赖加入find_package。
  
  ```
  # Do not just add this line to your CMakeLists.txt, modify the existing line
  find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    message_generation	#同时服务于msg,srv
  )
  ```
  
  ​		(2): 修改add_service_files
  
  ```
  add_service_files(
    FILES
    AddTwoInts.srv
  )
  ```

#### 4. 重新编译

添加完msg和srv后需要重新编译

```
# 需要在工作空间里使用catkin_make
$ roscd beginner_tutorials
$ cd ../..
$ catkin_make
$ cd -
```

### 2)服务Services

区别于Topic是一种单项的异步通信方式，Service通信是双向的，它不仅可以发送消息，同时还会有反馈。

所以service包括两部分：1.请求方clinet。2.服务提供方Server

通信机制：

​	服务提供方NodeB提供一个服务接口叫/Service。请求方NodeA向服务方发送一个请求request，服务方NodeA处理后反馈给请求方一个reply。

![](https://github.com/Fernweh-yang/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/service_structure.png?raw=true)

- rosservice相关命令

  - `rosservice list`：列出节点所用的所有服务

  - `rosservice info`：打印服务信息

  - `rosservice type [service]`：打印某一个服务的类型

    比如一个/clear 服务：`$ rosservice type /spawn`。在得到turtlesim/Spawn后可用`rossrv show turtlesim/Spawn`可以查看改类型的具体定义。也可直接：`$ rosservice type /spawn | rossrv show`

  - `rosservice call [service] [args]`：使用所提供的args调用服务

    比如调用/clear服务:`$ rosservice call /clear`

    比如调用/spawn服务：`$ rosservice call /spawn 2 2 0.2 ""`

  - `rosservice args`：打印服务参数

  - `rosservice find`：按服务类型查找服务

  - `rosservice uri`：打印服务ROSPRC uri

### 3)参数服务器parameter server

参数服务器是节点存储参数的地方、用于配置参数，全局共享参数。参数服务器使用互联网传输，在节点管理器中运行，实现整个通信过程。

相比于topic和service，参数服务器更加静态。它维护者一个数据字典，字典里存储着各种参数和配置。

**有三种参数服务器维护方式：**

1. 命令行维护

   使用rosparam语句来进行操作各种命令

   - `rosparam set param_key param_value`：设置参数

     - 设置背景rgb中r的值为150.`$ rosparam set /turtlesim/background_r 150`

   - `rosparam get param_key`：显示参数

     - 比如rgb中的g值`$ rosparam get /turtlesim/background_g`
     - 得到参数服务器中所有参数的值`$ rosparam get /`

   - `rosparam load file_name`：从文件加载参数

     - 将参数加载到一个新的命名空间`copy_turtle``$ rosparam load params.yaml copy_turtle`

   - `rosparam dump file_name`：保存参数到文件

     - 将所有的参数存入params,yaml文件`$ rosparam dump params.yaml`

   - `rosparam delete`：删除参数

   - `rosparam list`：列出所有参数名称

   - 上面load,dump命令用到的文件是yaml文件：如

     ```yaml
     name:'Zhangsan'
     age:20
     gender:'M'
     score{Chinese:80,Math:90}
     score_history:[85,82,88,90]
     ```

2. launch文件内读写

   launch文件中有很多标签，而与参数服务器相关的标签只有两个，一个是`<param>`，另一个是`<rosparam>`

   - <param>： 一般只设置一个参数

     ```xml
     <param name="parking_x" type="double" value="100.0" />
     ```

     代码中需要：

     ```c++
     // launch中如果<param .../>在<node.../>中，那么这里的参数是私有参数
     // c++代码这里，用~表示这个参数是私有参数。
     // 从launch中读取私有参数~parking_x的值到parking_x。
     ros::param::get("~parking_x",parking_x);
     ```

   - <rosparam>：读取一个yaml文件或多个参数

     相当于用命令行维护：`rosparam load file_name`

     ```xml
     <!--Attributes:
     *** command="load|dump|delete" (optional, default=load)
     *** file="$(find pkg-name)/path/foo.yaml" (load or dump commands)
     *** param="param-name"
     *** ns="namespace" (optional)
     -->
     <rosparam command="load" file="$(find rosparam)/example.yaml" />
     <rosparam command="delete" param="my/param" />
     <rosparam param="a_list">  [1, 2, 3, 4]  </rosparam>
     <rosparam>
       a: 1
       b: 2
     </rosparam>
     <!--Attributes:
     *** subst_value=true|false (optional):
     Allows use of substitution args in the YAML text.
     下面这个标签会使用内置的$(find ...)来寻找yaml里的叫whitelist的string，并替换它的value为default
     -->
     <arg name="whitelist" default="[3, 2]"/>
     <rosparam param="whitelist" subst_value="True">
       $(arg whitelist)
     </rosparam>
     ```

3. node源码:

   利用api来对参数服务器进行操作

### 4)动作库Action

一些常见的[动作库](https://sychaichangkun.gitbooks.io/ros-tutorial-icourse163/content/chapter4/4.6.html)

类似service通信机制，actionlib也是一种请求响应机制的通信方式，actionlib主要弥补了service通信的一个不足，就是当机器人执行一个长时间的任务时，假如利用service通信方式，那么publisher会很长时间接受不到反馈的reply，致使通信受阻。

所以actionlib则可以比较适合实现长时间的通信过程，actionlib通信过程可以随时被查看过程进度，也可以终止请求，这样的一个特性，使得它在一些特别的机制中拥有很高的效率。

**原理**：

客户端会向服务器发送目标指令和取消动作指令,而服务器则可以给客户端发送实时的状态信息,结果信息,反馈信息等等,从而完成了service没法做到的部分.

![](https://github.com/Fernweh-yang/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/actionlib.png?raw=true)

![](https://github.com/Fernweh-yang/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/action_interface.png?raw=true)

**Action规范**

利用动作库进行请求响应，动作的内容格式应包含三个部分，目标、反馈、结果。

- 目标

  机器人执行一个动作，应该有明确的移动目标信息，包括一些参数的设定，方向、角度、速度等等。从而使机器人完成动作任务。

- 反馈

  在动作进行的过程中，应该有实时的状态信息反馈给服务器的实施者，告诉实施者动作完成的状态，可以使实施者作出准确的判断去修正命令。

- 结果

  当运动完成时，动作服务器把本次运动的结果数据发送给客户端，使客户端得到本次动作的全部信息，例如可能包含机器人的运动时长，最终姿势等等。

- Action规范文件的后缀名是.action，内容格式如下：

  ```
  # Define the goal
  uint32 dishwasher_id  # Specify which dishwasher we want to use
  ---
  # Define the result
  uint32 total_dishes_cleaned
  ---
  # Define a feedback message
  float32 percent_complete
  ```

**编译需要修改CmakeLists.txt和package.xml**

- CmakeLists.txt:

  ```cmake
  find_package(catkin REQUIRED genmsg actionlib_msgs actionlib)
  
  add_action_files(DIRECTORY action FILES DoDishes.action) generate_messages(DEPENDENCIES actionlib_msgs)
  
  add_action_files(DIRECTORY action FILES Handling.action)
  
  generate_messages( DEPENDENCIES actionlib_msgs)
  ```

- package.xml

  ```xml
  <build_depend>actionlib </build_depend>
  <build_depend>actionlib_msgs</build_depend>
  <run_depend>actionlib</run_depend>
  <run_depend>actionlib_msgs</run_depend>
  ```

## $$时间Time和时长Duration

- 需要头文件`#include <ros/time.h>`和`#include <ros/duration.h>`

  time指的是某个时刻，duration指的是某个时段

  ```c++
  // 获取当前时间
  ros::Time begin=ros::Time::now();
  
  //定义类对象
  //_sec是秒，_nsec是纳秒
  ros::Time::Time(uint32_t _sec, uint32_t _nsec)
  ros::Time::Time(double t)
  ros::Duration::Duration(uint32_t _sec, uint32_t _nsec)
  ros::Duration::Duration(double t)
      
  //示例
  ros::Time time1(5,20000); 			//将实例time1时间定为5s+20000ns
  ros::Time start(ros::Time::now());  //将实例start时间定为当前时间
  ros:Duration one_hour(60*60,0);     //将实例one_hour时间段定为1小时
  double secs1=time1.toSec();         //将 Time类的实例 转为 double 型时间
  double secs2=one_hour.toSec();      //将 Duration类的实例 转为 double 型时间
  ```

- 睡眠sleep()和频率rate()

  - bool ros::Duration::sleep()

    ```c++
    //睡半秒
    ros::Duration(0.5).sleep();
    ```

  - ros::Rate

    ```c++
    // 以10hz的频率来执行while循环
    // 即循环程序运行时间不足0.1s时，会睡满0.1s
    ros::Rate r(10); // 10 hz
    while (ros::ok())
    {
      ... do some work ...
      r.sleep();
    }
    ```

- 定时器Timer

  ```c++
  #include "ros/ros.h"
  
  void callback(const ros::TimerEvent&)
  {
    ROS_INFO("Callback triggered");
  }
  
  int main(int argc, char **argv)
  {
    ros::init(argc, argv, "talker");
    ros::NodeHandle n;
    // Timer可以以一定的频率来循环的调用callback()回调函数。
    // 这里的频率是10hz,即0.1s
    ros::Timer timer1 = n.createTimer(ros::Duration(0.1), callback);
    ros::spin();
    return 0;
  }
  ```

- 时间和时长的运算

  - 1 hour + 1 hour = 2 hours (duration + duration = duration)
  - 2 hours - 1 hour = 1 hour (duration - duration = duration)
  - Today + 1 day = tomorrow (time + duration = time)
  - Today - tomorrow = -1 day (time - time = duration)
  - Today + tomorrow = error (time + time is undefined)



## $$调试(rqt_console)

- 使用`rqt_console`和`rqt_logger_level`   debug

  `rqt_concole`依附于ROS的logging framework来展现节点的输出

  `rqt_logger_level`允许改变在节点运行时改变他们的verbosity level详细等级(DEBUG,WARN,INFO and ERROR)

  在2个新终端分别启动：

  ```
  #新终端
  $ rosrun rqt_console rqt_console
  #新终端
  $ rosrun rqt_logger_level rqt_logger_level
  ```

  然后会出现两个window来显示报错等信息。

  最后再运行要debug的节点。

- logger level

  ```
  #按优先权高低排列：
  Fatal	有最高优先权
  Error	
  Warn
  Info
  Debug	有最低优先权
  # 设置为哪一个等级，我们就会得到它和更高等级的信息
  # 比如设置为Warn,则会得到Warn,Error,Fatal的消息
  ```

  

## $$编辑rosed

可以直接在一个包内直接用vim打开一个文件来编辑，而不需要具体的地址

使用：`$ rosed [package_name] [filename]`

例子：`$ rosed roscpp Logger.msg`

技巧：`$ rosed [package_name] <tab><tab>`

​			如果不知道具体的文件名，可以按两下tab显式这个包下所有的文件名。


## 五、写一个简单的Publisher 和 Subscriber

### 五(一)使用c++来写

#### 1. 写一个Publisher Node

- 创建一个cpp文件

  ```
  $ roscd beginner_tutorials	#来到要实现某功能的包
  $ mkdir -p src				#这个文件夹放所有包
  $ touch talker.cpp			#创建cpp文件
  ```

- 然后将[1.1](http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28c%2B%2B%29)的代码拷贝入talker.cpp
  
  需要做的事情：
  
  1. 初始化ROS系统
  2. 将关于chatter主题的std_msgs/String消息发送给master
  3. 每0.1秒循环发布一条消息到chatter
  
  ```c++
  #include "ros/ros.h"			//包含ROS系统所需要的大部分头文件
  #include "std_msgs/String.h"	//包含字符串类型的msg
  #include <sstream>
  
  int main(int argc, char **argv)
  {
    //初始化ros,允许通过命令行进行name remapping
    //也是定义我们节点名字的地方"talker"
    ros::init(argc, argv, "talker");	
    //创建当前进程节点的句柄Handle
    ros::NodeHandle n;	
  
   
    // 告诉master，将要发送一个std_msgs/String类型的message到话题chatter。
    // 这时master会让所有节点接听来自chatter将要发送的消息。
    // 第二个参数是发布队列的大小，如果发布的太快，会在丢弃旧消息前保存1000条。
    // NodeHandle::advertise()返回一个ros::Publisher对象。提供2个功能：
    //	1.包含一个publish()方法，可以发送消息到创建的topic上
    // 	2.当它超过范围，它会自动停止发送
    ros::Publisher chatter_pub = n.advertise<std_msgs::String>("chatter", 1000);
    
    // ros::Rate对象object 允许设置一个想要的循环频率，这里是0.1s/10hz
    // 追踪上次调用的ros::sleep()后过了多久来保证正确的睡眠时间
    // 发送消息的频率为10hz,一秒发10条消息，周期为1/10s
    ros::Rate loop_rate(10);
  
  
    // roscpp文件自带SIGINT,可以提供诸如ctrl+c这样的信号。
    // ros::ok()如下情况会返回false：
    //	1.收到一个SIGINT信号
    //	2.被另一个同名的节点踢出了网络
    //	3.ros::shutdown()被程序的其他部分调用
    //	4.所有的ros::NodeHandles被销毁了  
    int count = 0;
    while (ros::ok())
    {
  
      std_msgs::String msg;
      std::stringstream ss;
      ss << "hello world " << count;
      msg.data = ss.str();
  	
      //可以在终端打印该消息
      ROS_INFO("%s", msg.data.c_str());
      //将消息发送给master,所有的subscriber会接收到这个msg
      chatter_pub.publish(msg);
  	
      // 用于调用callback()函数
      // subscriber接收的消息并不是立刻就调用回调函数的，只有等到spin()或spinOnce()执行的时候才会被调用
      // spin()在调用后，不再运行后面的程序，所以一般不出现在循环中。仅仅知识相应topic时使用。
      // spinOnce()在调用后，还会继续执行后面的程序。除了调用回调函数还要做其他重复工作时使用。
      ros::spinOnce();	
       
      // 靠sleep()函数来保证连续2个信号之间的时间恰好为一个周期
      loop_rate.sleep();	
      ++count;
    }
  
  
    return 0;
  }
  ```
  

#### 2. 写一个Subscriber Node

- 创建一个cpp文件

  ```
  touch listener.cpp
  ```

- 将[2.1](http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28c%2B%2B%29)的代码复制过去

  需要做的事情

  1. 初始化ros系统
  2. subscribe订阅chatter话题
  3. spin()循环，等待消息的到达
  4. 当消息到达，调用chatterCallback

  ```c++
  #include "ros/ros.h"
  #include "std_msgs/String.h"
  
  
  // 这是一个Callback函数，当chatter话题收到一个新message的时候会被调用。
  // 这些消息在boost shared_ptr中传递，所有我们可以根据需要将他存储起来。
  void chatterCallback(const std_msgs::String::ConstPtr& msg)
  {
    ROS_INFO("I heard: [%s]", msg->data.c_str());
  }
  
  int main(int argc, char **argv)
  {
  
    ros::init(argc, argv, "listener");
    ros::NodeHandle n;
    
    
      
    // 从master那订阅chatter话题。ROS会在每次有新消息到达时，调用chatterCallback()函数。
    // 第二个参数是消息队列的大小，以防我们处理消息的速度不够快时可以先把消息保存起来。超过1000条消息的时候会在新消息到达时删除旧消息。
    // NodeHandle::subscribe()返回一个ros::Subscriber对象。这个对象必须被hold on保留，直到我们不想订阅为止。
    ros::Subscriber sub = n.subscribe("chatter", 1000, chatterCallback);
  
    // ros::spin()会进入一个循环，并尽可能块的调用message callbacks.
    // 当ros::ok()返回错误的时候，ros::spin()也会跟着自动停止。
    // 用于调用callback()函数
    // 用于调用callback()函数
    // subscriber接收的消息并不是立刻就调用回调函数的，只有等到spin()或spinOnce()执行的时候才会被调用
    // spin()在调用后，不再运行后面的程序，所以一般不出现在循环中。仅仅知识相应topic时使用。
    // spinOnce()在调用后，还会继续执行后面的程序。除了调用回调函数还要做其他重复工作时使用。
    ros::spin();
  
    return 0;
  }
  ```

#### 3. 构建building上面写好的节点

- 打开CMakeLists.txt

  ```
  $ source ./catkin_ws/devel/setup.bash
  #将包加入ros环境后使用rosed
  $ rosed beginner_tutorials CMakeLists.txt
  ```

- 编辑CMakeLists.txt

  将如下代码加入文件末尾

  ```
  add_executable(talker src/talker.cpp)
  # 作为Groovy常规,可以使用下面的变量去依赖所有必要的目标
  target_link_libraries(talker ${catkin_LIBRARIES})
  # 确保包的消息头文件在使用之前先被调用。如果消息来自工作空间内的其他包，需要给他们各自的
  # generation targets添加依赖项，因为catkin会并行的编译所有projects.
  add_dependencies(talker beginner_tutorials_generate_messages_cpp)
  
  add_executable(listener src/listener.cpp)
  target_link_libraries(listener ${catkin_LIBRARIES})
  add_dependencies(listener beginner_tutorials_generate_messages_cpp)
  ```

- 运行catkin_make

  ```
  # In your catkin workspace
  $ cd ~/catkin_ws
  $ catkin_make  
  ```

### 五(二)使用python来写

#### 1. 写一个Publisher Node

- 创建文件

  ```shell
  $ roscd beginner_tutorials
  $ mkdir scripts				##scripts文件夹放py脚本，src文件夹放c++程序。
  $ cd scripts
  ##教程上的publisher下下来
  $ wget https://raw.github.com/ros/ros_tutorials/kinetic-devel/rospy_tutorials/001_talker_listener/talker.py
  $ chmod +x talker.py		##给talker.py文件加权限，x表示可执行
  ```

- 代码

  ```python
  #!/usr/bin/env python				每个ROS节点都会有这么一行，保证脚本是被用作python script
  # license removed for brevity
  import rospy						
  from std_msgs.msg import String		
  
  def talker():
      #说明这个节点是用来给chatter话题传输String类型的消息的。queue_size是当subsriber接收速度不够快时，排队消息的数量。
      pub = rospy.Publisher('chatter', String, queue_size=10)
      #初始化，且定义该节点名为talker.
      #anonymous=True通过在talker后加随机数来确保节点名是唯一的，因为一个ROS系统中不能出现同名节点，否则会被刷掉。
      rospy.init_node('talker', anonymous=True)
      rate = rospy.Rate(10) # 10hz
      #rospy.is_shutdown()用于判断是否存在需要停止程序的情况，比如ctrl+c。
      while not rospy.is_shutdown():
          hello_str = "hello world %s" % rospy.get_time()
  		"""
  		rospy.loginfo(str)具有3个功能triple-duty
  		1. 将消息输出到屏幕
  		2. 将消息写入Node's log file
  		3. 将消息写给rosout.rosout是一个handy方便的debug工具
  		"""
          rospy.loginfo(hello_str)
          #发布消息给chatter话题
          pub.publish(hello_str)
          rate.sleep()
  
  if __name__ == '__main__':
      try:
          talker()
      except rospy.ROSInterruptException:
          pass
  ```

  - 并把py脚本加入CMakeLists.txt,以保证脚本正确的被加载和编译

    用`rosed beginner_tutorials CMakeLists.txt`打开后加入：

    ```
    catkin_install_python(
    	PROGRAMS scripts/talker.py
      	DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    )
    ```

#### 2. 写一个Subscriber Node

- 创建一个py文件

  这里直接从教材下载一个subscriber

  ```shell
  $ roscd beginner_tutorials/scripts/
  $ wget https://raw.github.com/ros/ros_tutorials/kinetic-devel/rospy_tutorials/001_talker_listener/listener.py
  $ chmod +x listener.py
  ```

- 代码：

  ```python
  #!/usr/bin/env python
  import rospy
  from std_msgs.msg import String
  
  def callback(data):
      rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
      
  def listener():
  
      # In ROS, nodes are uniquely named. If two nodes with the same
      # name are launched, the previous one is kicked off. The
      # anonymous=True flag means that rospy will choose a unique
      # name for our 'listener' node so that multiple listeners can
      # run simultaneously.
      rospy.init_node('listener', anonymous=True)
  	
      # 当subscriber从chatter话题那收到string类型的消息时，会调用callback
      rospy.Subscriber("chatter", String, callback)
  
      # spin() simply keeps python from exiting until this node is stopped
      rospy.spin()
  
  if __name__ == '__main__':
      listener()
  ```

  - 同样将subsrciber加入到CMakeLists.txt

    ```
    catkin_install_python(
    	PROGRAMS scripts/talker.py scripts/listener.py
     	DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    )
    ```

#### 3. building上面写好的节点

```
$ cd ~/Desktop/catkin_ws
$ catkin_make
```

### 五(三)测试Publisher和Subscriber

1. `$ roscore`

2. 每开一个终端都要`$ source ./devel/setup.bash `

   也可以直接在~/.bashrc中加入想要用的工作空间的/devel/setup.bash

3. 新终端中运行发布者

   ```
   $ rosrun beginner_tutorials talker      (C++)
   $ rosrun beginner_tutorials talker.py   (Python) 
   ```

4. 新终端中运行接收者

   ```
   $ rosrun beginner_tutorials listener     (C++)
   $ rosrun beginner_tutorials listener.py  (Python) 
   ```



## 六、写一个简单的Service和Client

### 六(一)使用C++来写

#### 1. 写一个Service Node

​	功能：接收2个ints后返回他们的和

- 创建一个cpp文件

  ```shell
  $ cd ~/Desktop/catkin_ws/src/beginner_tutorials/src
  $ touch add_two_ints_server.cpp
  ```

- 将[1.1](http://wiki.ros.org/ROS/Tutorials/WritingServiceClient%28c%2B%2B%29)的代码复制过来

  这里头文件#include "beginner_tutorials/AddTwoInts.h"在**[四、创建ROS msg和srv](http://wiki.ros.org/ROS/Tutorials/CreatingMsgAndSrv#Creating_a_srv)**时创建

  ```c++
  #include "ros/ros.h"
  #include "beginner_tutorials/AddTwoInts.h"
  
  //该函数提供了两个整数相加的服务，它接收 srv 文件中定义的请求和响应类型，并返回一个布尔值。
  bool add(beginner_tutorials::AddTwoInts::Request  &req,
           beginner_tutorials::AddTwoInts::Response &res)
  {
    res.sum = req.a + req.b;
    ROS_INFO("request: x=%ld, y=%ld", (long int)req.a, (long int)req.b);
    ROS_INFO("sending back response: [%ld]", (long int)res.sum);
    return true;
  }
  
  int main(int argc, char **argv)
  {
    ros::init(argc, argv, "add_two_ints_server");
    ros::NodeHandle n;
    //创建一个服务add_two_ints，用于广播advertise服务。接收到请求request后会调用add方法
    ros::ServiceServer service = n.advertiseService("add_two_ints", add);
    ROS_INFO("Ready to add two ints.");
    ros::spin();
  
    return 0;
  }
  ```

#### 2. 写一个Client Node

- 创建一个cpp文件

  ```
  $ cd ~/Desktop/catkin_ws/src/beginner_tutorials/src
  $ touch add_two_ints_client.cpp
  ```

- 将[2.1](http://wiki.ros.org/ROS/Tutorials/WritingServiceClient%28c%2B%2B%29)的代码复制过来

  ```c++
  #include "ros/ros.h"
  #include "beginner_tutorials/AddTwoInts.h"
  #include <cstdlib>
  
  int main(int argc, char **argv)
  {
    ros::init(argc, argv, "add_two_ints_client");
    if (argc != 3)
    {
      ROS_INFO("usage: add_two_ints_client X Y");
      return 1;
    }
  
    ros::NodeHandle n;
    /*
    创建一个add_two_ints service服务的client客户
    ros::ServiceClient对象用来呼叫服务
    */
    ros::ServiceClient client = n.serviceClient<beginner_tutorials::AddTwoInts>("add_two_ints");
    //instantiate实例化一个autogenerated service class.
    beginner_tutorials::AddTwoInts srv;
    //service类包含2个成员request和response。AddTwoInts服务类有2个request值    
    srv.request.a = atoll(argv[1]);
    srv.request.b = atoll(argv[2]);
      
    //client.call(srv)呼叫服务。如果成功返回true,并返回response值。如果失败返回false,此时response值无意义。
    if (client.call(srv))
    {
      //AddTwoInts服务类有1个reponse值 
      ROS_INFO("Sum: %ld", (long int)srv.response.sum);
    }
    else
    {
      ROS_ERROR("Failed to call service add_two_ints");
      return 1;
    }
  
    return 0;
  }
  ```
  

#### 3. 构建节点并运行

- 将server和client加入CMakeLists.txt

  加在末尾：

  ```
  add_executable(add_two_ints_server src/add_two_ints_server.cpp)
  target_link_libraries(add_two_ints_server ${catkin_LIBRARIES})
  add_dependencies(add_two_ints_server beginner_tutorials_gencpp)
  
  add_executable(add_two_ints_client src/add_two_ints_client.cpp)
  target_link_libraries(add_two_ints_client ${catkin_LIBRARIES})
  add_dependencies(add_two_ints_client beginner_tutorials_gencpp)
  ```

- 编译

  ```
  # In your catkin workspace
  cd ~/Desktop/catkin_ws
  catkin_make
  ```

- 运行
  - `roscore`
  - 新终端打开服务`rosrun beginner_tutorials add_two_ints_server`
  - 新终端打开客户`$ rosrun beginner_tutorials add_two_ints_client 1 3`

### 六(二)使用python来写

# ROS进阶

## #、Autonomous Systems课程相关:

### Lab2:Task2

- 运用tf，进行坐标系frame转换

```c++
#include <ros/ros.h>
#include <ros/console.h>
#include <tf/transform_broadcaster.h>
#include <iostream>
#include <cmath>

class FramesPublisherNode{
 //创建节点句柄nh
 ros::NodeHandle nh;
 ros::Time startup_time;

 ros::Timer heartbeat;
 //1.定义2个广播broadcaster br1和br2
 tf::TransformBroadcaster br1;	
 tf::TransformBroadcaster br2;

 public:
  FramesPublisherNode(){
    // NOTE: This method is run once, when the node is launched.
    startup_time = ros::Time::now();
    heartbeat = nh.createTimer(ros::Duration(0.02), &FramesPublisherNode::onPublish, this);
    heartbeat.start();
  }

  void onPublish(const ros::TimerEvent&){
    ros::Duration E_time = ros::Time::now() - startup_time;
    double time = E_time.toSec();

    //2. 声明2个变量用来分别存储2个无人机的转换信息
    tf::Transform AV1World(tf::Transform::getIdentity());
    tf::Transform AV2World(tf::Transform::getIdentity());

    //3. 设置坐标转换，（1.，2.，3.）为子坐标系av1在父坐标系world坐标系中的坐标，
    AV1World.setOrigin(tf::Vector3(std::cos(time), std::sin(time), 0));
    AV2World.setOrigin(tf::Vector3(std::sin(time), 0, std::cos(2*time)));
	
    // 4.1 定义无人机1号的旋转
    tf::Quaternion q1;
    q1.setRPY(0,0,time);//（0，0，time）av1坐标系在world坐标下的roll(绕X轴)，pitch(绕Y轴)，yaw(绕Z轴) 的旋转度数.
    AV1World.setRotation(q1);
      
    // 4.2 定义无人机2号的旋转
    tf::Quaternion q2;
    q2.setRPY(0,0,0);//（0，0，0）av2在world坐标系下的roll(绕X轴)，pitch(绕Y轴)，yaw(绕Z轴) 的旋转度数，现在都是0度
    AV2World.setRotation(q2);

	//将变换广播出去，发布了world和无人机1号av1，2号av2之间的坐标关系
    br1.sendTransform(tf::StampedTransform(AV1World, ros::Time::now(),"world", "av1"));
    br2.sendTransform(tf::StampedTransform(AV2World, ros::Time::now(),"world", "av2"));
  }
};

int main(int argc, char** argv){
  //初始化ROS，节点名为frames_publisher_node
  ros::init(argc, argv, "frames_publisher_node");
  FramesPublisherNode node;
  ros::spin();
  return 0;
}

```



## #、ROS中TF的使用

- 简介：

  - TF(TranssForm)坐标转换，包括位置和姿态2个方面。**要注意区分坐标转换和坐标系frame的转换**

    `坐标转换是一个坐标在不同坐标系下的表示，而坐标系转换不同坐标系的相对位姿关系`

  - tf是一个树状结构，维护坐标系之间的关系，靠**话题通信机制**来持续地发布不同link(比如手部、头部、某关节)之间的坐标关系。

    作为树状结构，要保证父子坐标系都有某个节点在持续地发布他们之间的位姿关系，才能使树状结构保持完整。只有父子坐标系的位姿关系能被正确的发布，才能保证任意两个frame之间的连通。

  - 每两个相邻frame之间靠节点发布它们之间的位姿关系，这种节点称为**broadcaster**。broadcaster就是一个发布器publisher,如果两个frame之间发生了相对运动，broadcaster就会发布相关消息。

### 1. 用c++写一个tf broadcaster(subscriber)

- 创建包

  ```
   $ cd %YOUR_CATKIN_WORKSPACE_HOME%/src
   $ catkin_create_pkg learning_tf tf roscpp rospy turtlesim
   
   $ cd ..
   $ catkin_make
   $ source ./devel/setup.bash
  ```

- 创建一个.cpp文件

  ```
  $ roscd learning_tf
  $ cd src
  $ touch turtle_tf_broadcaster.cpp
  ```

  前往包下src处创建.cpp文件并输入:

  ```c++
  #include <ros/ros.h>
  #include <tf/transform_broadcaster.h>
  #include <turtlesim/Pose.h>
  
  std::string turtle_name;
  
  void poseCallback(const turtlesim::PoseConstPtr& msg){
    //创建一个TransformBroadcaster实例，用于发送transformations转换
    static tf::TransformBroadcaster br;
    //创建一个Transform实例，存储转换信息
    tf::Transform transform;
    //将小乌龟的初始坐标设置为坐标原点。小乌龟坐标是2维的，所以这里z是0.0
    transform.setOrigin( tf::Vector3(msg->x, msg->y, 0.0) );
    //创建一个Quaternion实例，用于存储坐标系绕各轴旋转的角度
    tf::Quaternion q;
    //roll(绕X轴)，pitch(绕Y轴)，yaw(绕Z轴) 的旋转度数
    q.setRPY(0, 0, msg->theta);
    transform.setRotation(q);
    //将坐标系转换发送出去。第1个参数:转换信息。2：给正在发布的转换一个时间戳。3：父坐标的名称。4：子坐标的名称
    br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", turtle_name));
  }
  
  int main(int argc, char** argv){
    ros::init(argc, argv, "my_tf_broadcaster");
    if (argc != 2){ROS_ERROR("need turtle name as argument"); return -1;};
    turtle_name = argv[1];
  
    ros::NodeHandle node;
    ros::Subscriber sub = node.subscribe(turtle_name+"/pose", 10, &poseCallback);
  
    ros::spin();
    return 0;
  };
  ```

- 运行

  1. 用`rosed learning_tf CMakeLists.txt`打开编译文件

     末尾加入：

     ```
     add_executable(turtle_tf_broadcaster src/turtle_tf_broadcaster.cpp)
     target_link_libraries(turtle_tf_broadcaster ${catkin_LIBRARIES})
     ```

  2. 工作空间目录下编译

     `$ catkin_make`

  3. 在learn_ws/src/learning_tf/launch下创建.launch文件

     ```xml
     <launch>
         <!-- Turtlesim Node-->
         <node pkg="turtlesim" type="turtlesim_node" name="sim"/>
     
         <node pkg="turtlesim" type="turtle_teleop_key" name="teleop" output="screen"/>
         <!-- Axes -->
         <param name="scale_linear" value="2" type="double"/>
         <param name="scale_angular" value="2" type="double"/>
     
         <node pkg="learning_tf" type="turtle_tf_broadcaster"
               args="/turtle1" name="turtle1_tf_broadcaster" />
         <node pkg="learning_tf" type="turtle_tf_broadcaster"
               args="/turtle2" name="turtle2_tf_broadcaster" />
     
     </launch>
     ```
     
  4. 运行:

     ` $ roslaunch learning_tf start_demo.launch`

   5. 新终端中检查结果

      ` $ rosrun tf tf_echo /world /turtle1`

### 2. 用c++写一个tf listener(client)

- 创建包：

  ```shell
   $ roscd learning_tf
   $ cd src
   $ touch turtle_tf_listener.cpp
  ```

- 复制[1.1](http://wiki.ros.org/tf/Tutorials/Writing%20a%20tf%20listener%20%28C%2B%2B%29)如下代码

  ```c++
  #include <ros/ros.h>
  #include <tf/transform_listener.h>
  #include <geometry_msgs/Twist.h>
  #include <turtlesim/Spawn.h>
  
  int main(int argc, char** argv){
    ros::init(argc, argv, "my_tf_listener");
  
    ros::NodeHandle node;
    
    //等待一个叫spawn的服务可用。
    ros::service::waitForService("spawn");
    //创建一个服务spawn的客户add_turtle
    ros::ServiceClient add_turtle = node.serviceClient<turtlesim::Spawn>("spawn");
    //实例化一个自动生成的服务类。参见创建ROS msg和srv。
    turtlesim::Spawn srv;
    //呼叫服务
    add_turtle.call(srv);
  	
    //创建一个Publisher实例，将消息发送到给话题cmd_vel
    ros::Publisher turtle_vel = node.advertise<geometry_msgs::Twist>("turtle2/cmd_vel", 10);
    
    //实例化一个TransformListener对象。一旦被建立，就会开始接收tf transformations
    tf::TransformListener listener;
  
    ros::Rate rate(10.0);
    while (node.ok()){
      tf::StampedTransform transform;
      try{
        /*
  	  向listener请求一个特定转换
  	  1.想要从坐标/turtle1到/turtle2的转换
  	  2.我们想什么时候进行转换。Time(0)最近一个可以进行转换的时刻。
  	  3.transform储存产生的转换
  	  
  	  参照笔记 3.添加一个坐标frame
  	  如果把/turtle1改为/carrot1，这时乌龟2号就开始追着坐标系carrot1跑，而不是追着乌龟1号跑了
        */
        listener.lookupTransform("/turtle2", "/turtle1",
                                 ros::Time(0), transform);
      }
      catch (tf::TransformException &ex) {
        ROS_ERROR("%s",ex.what());
        ros::Duration(1.0).sleep();
        continue;
      }
  	
      //实例化一个消息类。参见创建ROS msg和srv
      geometry_msgs::Twist vel_msg;
      vel_msg.angular.z = 4.0 * atan2(transform.getOrigin().y(),
                                      transform.getOrigin().x());
      vel_msg.linear.x = 0.5 * sqrt(pow(transform.getOrigin().x(), 2) +
                                    pow(transform.getOrigin().y(), 2));
      turtle_vel.publish(vel_msg);
  
      rate.sleep();
    }
    return 0;
  };
  ```

- 运行：

  1. `rosed learning_ws CMakeLists.txt`

     加到最后

     ```
     add_executable(turtle_tf_listener src/turtle_tf_listener.cpp)
     target_link_libraries(turtle_tf_listener ${catkin_LIBRARIES})
     ```

  2. 编译`$ catkin_make`

     这会在devel/lib/learning_tf文件夹生成一个二进制文件turtle_tf_listener.

  3. 将这个二进制文件加入start_demo.launch

     ```xml
     <launch>
     ...
     <node pkg="learning_tf" type="turtle_tf_listener"
           name="listener" />
     </launch>
     ```

  4. 运行.launch文件

     ```
      $ roslaunch learning_tf start_demo.launch
     ```

### 3. 用c++添加一个坐标frame

- 创建文件

  ```shell
  $ touch ~/Desktop/learn_ws/src/learning_tf/src frame_tf_broadcaster.cpp
  ```
  
- 写入代码

  ```c++
  #include <ros/ros.h>
  #include <tf/transform_broadcaster.h>
  
  int main(int argc, char** argv){
    ros::init(argc, argv, "my_tf_broadcaster");
    ros::NodeHandle node;
  
    tf::TransformBroadcaster br;
    tf::Transform transform;
  
    ros::Rate rate(10.0);
    while (node.ok()){
      /*
      设置坐标系位置转换.子坐标carrot1在父坐标turtle1的右边2单位
      这时我们创建的子坐标相对父坐标的位置是不随时间变化的，如果想要让它随时间变化：
      transform.setOrigin( tf::Vector3(2.0*sin(ros::Time::now().toSec()), 2.0*cos(ros::Time::now().toSec()), 0.0) );
      */
      transform.setOrigin( tf::Vector3(0.0, 2.0, 0.0) );
      //设置坐标系角度转换。Quaternion四元数
      transform.setRotation( tf::Quaternion(0, 0, 0, 1) );
      //发送转换。从父坐标turtle1转到子坐标carrot1.
      br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "turtle1", "carrot1"));
      rate.sleep();
    }
    return 0;
  };
  ```
  
- 运行
  
  1. 加到`rosed learning_ws CMakeLists.txt`底部
  
     ```
     add_executable(frame_tf_broadcaster src/frame_tf_broadcaster.cpp)
     target_link_libraries(frame_tf_broadcaster ${catkin_LIBRARIES})
     ```
  
  2. 编译`catkin_make`
  
  3. 加入`rosed learning_ws start_demo.launch`
  
     ```xml
     <launch>
         ...
         <node pkg="learning_tf" type="frame_tf_broadcaster"
               name="broadcaster_frame" />
     </launch>
     ```
  
  4. 运行` $ roslaunch learning_tf start_demo.launch`
  

### 4.tf和time

- 问题：想要在特定时刻进行转移。

  在笔记2.xx中客户**`src/turtle_tf_listener.cpp`**：
  
  ```c++
  try{
    listener.lookupTransform("/turtle2", "/turtle1",  
	                           ros::Time(0), transform);
  ```

	这里用的Time(0)是最近一个可以进行转动的时刻，如果想在特定时刻，比如立刻马上：
	
	```c++
	try{
		listener.lookupTransform("/turtle2", "/turtle1",  
	                         ros::Time::now(), transform);
	```

​		但这里会出现问题，因为每一个listener有一个buffer缓冲器用来存储来自所有tf broadcaster的坐标转换信息。但每一个broadcaster将信息发过来，都需要几毫秒的时间。所以这里用Time::now()立刻进行转移，就会发生在收到转换信息之前，从而出错。
- 解决：使用waitForTransform

  先wait然后look up是否有转换信息传过来。2个函数必须用同一个Time实例now，不然时间戳对不上。

  ```c++
  try{
      ros::Time now = ros::Time::now();
      listener.waitForTransform("/turtle2", "/turtle1",
                                now, ros::Duration(3.0));
      listener.lookupTransform("/turtle2", "/turtle1",
                               now, transform);
  ```
  
  - waitForTransform四个参数：
  
    1. 等待从这个frame: /turtle2
    2. 转到这个frame:     /turtle1
    3. 在这个时间点
    4. 最大等待的时间
### 5.用tf进行time travel

- 问题：让乌龟2到乌龟1 五秒之前在的地方。

  同样修改**`src/turtle_tf_listener.cpp`**：

  ```c++
    try{
      ros::Time now = ros::Time::now();
      ros::Time past = now - ros::Duration(5.0);
      listener.waitForTransform("/turtle2", now,
                                "/turtle1", past,
                                "/world", ros::Duration(1.0));
      listener.lookupTransform("/turtle2", now,
                               "/turtle1", past,
                               "/world", transform);
  ```

  - 此时的lookupTransform有6个参数
    1. 本次转移从该坐标frame
    2. 转移的时间
    3. 要转移到的目标坐标frame
    4. 什么时候的目标坐标位置
    5. 一个不随时间变化的参考坐标
    6. 存储所有的转换信息

  

## #、读取YAML文件

- **在包内创建一个YAML文件**

  包的结构

  ```
  /include          : Cpp headers
  /src              : Cpp code
  /scripts          : Python code
  /config           : YAML files
  /launch           : Launch files
  CMakeLists.txt
  package.xml
  ```

- **编写yaml文件**

  ```yaml
  text: "Hello"
  number_int: 42 
  number_float: 21.3
  enable_boolean: true
  list_of_elements:
      - 1
      - 2
      - 3
      - 4
  dictionary: {
      another_text: "World",
      another_number: 12,
  }
  ```

- **从yaml文件下载参数params到Parameter Server**

  - 用命令行`rosparam load my_params.yaml`下载

    可以用`rosparam list`查看parameter里已有的键(属性)

    可以用`rosparam dump dump.yaml`将Parameter Server里的键值对反向下载到yaml文件里

  - 用launchfile来下载

    ```xml
    <launch>
        <rosparam file="$(find my_custom_package)/config/my_params.yaml" />
    </launch>
    
    #最后结果类似
     * /list_of_anything: [1, 2, 3, 4]
     * /number_float: 21.3
     * /number_int: 42
    ```
    
    - 可以直接用find命令来找到相应包下的那个yaml文件
    
  - 如果想要给所有的参数增加一个prefix前缀
    
    ```xml
    <launch>
        <group ns="custom_prefix">
            <rosparam file="$(find my_custom_package)/config/my_params.yaml" />
        </group>
    </launch>
    
    #最后结果类似
     * /custom_prefix/list_of_anything: [1, 2, 3, 4]
     * /custom_prefix/number_float: 21.3
     * /custom_prefix/number_int: 42
    ```
  
- **在代码中使用这些参数**
  
  比如要用上面例子中的`* /custom_prefix/number_float: 21.3`这一参数
  
  - python:
  
    `rospy.get_param("/custom_prefix/number_float")`
  
  - C++:
  
    ```
    ros::NodeHandle nh;
    double number_to_get;
    nh.getParam("/custom_prefix/number_float", number_to_get);
    ```
  
    
  
    
  
  
  
  
  
  
  
  

