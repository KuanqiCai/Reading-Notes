# ROS基础

## 一、创建工作空间Workspace

```shell
## 创建工作空间
$ mkdir -p ~/catkin_ws/src
$ cd ~/catkin_ws/
$ catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
## source the new setup.*sh file
$ source devel/setup.bash
## 检查环境变量是否包含了当前工作空间
$ echo $ROS_PACKAGE_PATH
```

## #、ROS文件系统Filesystem Concepts

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

    ​			`$ roscd roscpp/cmake  `前往roscpp包所在的目录中cmake子目录。

  - `roscd log`如果运行过ros程序后，可以查看日志文件

- **rosls**

  相当于ls，直接显式包里的内容

  ​	用法：`$ rosls <package-or-stack>[/subdir]`

  ​	例子：`rosls roscpp_tutorials`

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

  

 ## #、节点node

- 一些概念
  - [Nodes](http://wiki.ros.org/Nodes): A node is an executable that uses ROS to communicate with other nodes.
  - [Messages](http://wiki.ros.org/Messages): ROS data type used when subscribing or publishing to a topic.
  - [Topics](http://wiki.ros.org/Topics): Nodes can *publish* messages to a topic as well as *subscribe* to a topic to receive messages.
  - [Master](http://wiki.ros.org/Master): Name service for ROS (i.e. helps nodes find each other)
  - [rosout](http://wiki.ros.org/rosout): ROS equivalent of stdout/stderr
  - [roscore](http://wiki.ros.org/roscore): Master(provides name service for ROS) + rosout + parameter server 
- `roscore`
  - 使用ROS第一步就是输入`roscore`

- `rosnode`

  - `rosnode list`查看当前运行着的节点

    这时候返回：`/rosout`

  - `rosnode info /rosout`返回特定节点的信息

- `rosrun`

  - `rosrun [package_name] [node_name]`直接运行某一个包下的某节点

    - 例子：`$ rosrun turtlesim turtlesim_node`

      ​			`$ rosrun turtlesim turtlesim_node __name:=my_turtle`还能reassign名

## #、话题topic

- 小案例

  - 输入`$ roscore`
  - 新终端运行节点`$ rosrun turtlesim turtlesim_node`
  - 新终端运行节点`$ rosrun turtlesim turtle_teleop_key`

- 话题topics：

  上述案例中2个节点`turtlesim_node`和`turtle_teleop_key`就是通过话题topic来实施通信的：

  1. 发布者Publisher:`turtlesim_node`  **publishing** key strokes键给topic。

  2. 接收者Subscriber:`turtle_teleop_key` 从topic那里**subscribes** 这个keystrokes。

  - `rqt_graph`

    用于展示各个节点之间的关系

    新终端输入`$ rosrun rqt_graph rqt_graph`可看到两个节点通过话题`/turtle1/cmd_vel`联系在一起

  - `rostopic`

    用于获取ros话题的消息。可用`$ rostopic -h`查看帮助文档

    - `rostopic echo /topic`

      1.新终端输入`rostopic echo /turtle1/cmd_vel` 

      2.再返回`turtle_teleop_key`的终端进行控制。`rostopic`终端就会出现传输给话题的消息。

    - `rostopic list /topic`

      返回所有当前订阅和发布的话题的列表，可用`$ rostopic list -h`查看所需参数

- ROS Messages消息

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

## #、ROS Services和Parameters

- Services:

  服务是另一种节点通信方式，允许节点发送request请求和接收响应。

- `Rosservis`

  提供了很多命令，能被用于ROS 的 client/service 框架

  ```
  rosservice list         print information about active services
  rosservice call         call the service with the provided args
  rosservice type         print service type
  rosservice find         find services by service type
  rosservice uri          print service ROSRPC uri
  ```

  - `$ rosservice list`

    列出节点所用的所有服务

  - `$ rosservice type [service]`

    显式某一个服务的类型，比如一个/clear 服务：`$ rosservice type /spawn`

    - 得到turtlesim/Spawn后可用`rossrv show turtlesim/Spawn`可以查看改类型的具体定义

      也可直接：`$ rosservice type /spawn | rossrv show`

  - `$ rosservice call [service] [args]`

    调用某个服务，比如调用/clear服务:`$ rosservice call /clear`

    ​						   比如调用/spawn服务：`$ rosservice call /spawn 2 2 0.2 ""`

- `Rosparam`

  用于存储和manipulate操作Parameter Server参数服务器的数据。

  ```
  rosparam set            set parameter
  rosparam get            get parameter
  rosparam load           load parameters from file
  rosparam dump           dump parameters to file
  rosparam delete         delete parameter
  rosparam list           list parameter names
  ```

  - `$ rosparam list`

    查看所有节点所有的参数
    
  - `rosparam set [param_name]`和`rosparam get [param_name]`

    - 设置背景rgb中r的值为150.

      `$ rosparam set /turtlesim/background_r 150`

      使参数变化起效果

      `$ rosservice call /ckear`

    - 得到某一个参数的值，比如rgb中的g值

      `$ rosparam get /turtlesim/background_g `

      得到参数服务器中所有参数的值

      `$ rosparam get /`

  - `rosparam dump [file_name] [namespace]`和`rosparam load [file_name] [namespace]`

    - 将所有的参数存入params,yaml文件

      `$ rosparam dump params.yaml`

    - 将参数导入一个新的命名空间`copy_turtle`

      `$ rosparam load params.yaml copy_turtle`

## #、调试(rqt_console)和启动文件(roslaunch)

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

- 使用roslaunch同时打开多个节点

  用法：`$ roslaunch [package] [filename.launch]`

  1. 回到工作空间`$ cd ~/catkin_ws`

  2. 将当前环境导入ROS环境`$ cd ~/catkin_ws`

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

## #、编辑rosed

可以直接在一个包内直接用vim打开一个文件来编辑，而不需要具体的地址

使用：`$ rosed [package_name] [filename]`

例子：`$ rosed roscpp Logger.msg`

技巧：`$ rosed [package_name] <tab><tab>`

​			如果不知道具体的文件名，可以按两下tab显式这个包下所有的文件名。

## 四、创建ROS msg和srv

### 1.msg,srv简介

- **msg**: msg files是描述the fields of a ROS message的简单文本文件。他们以不同的语言来生成messages的源代码。储存在msg directory目录。

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

    

- **srv**: srv files描述了一个服务，包含2个部分：request 和 response。储存在srv directory。

  - src也由一行行field type 和 field name组成。request和response由"---"隔开。

    ```
    int64 A
    int64 B
    ---
    int64 Sum
    ```

    A,B是request,Sum是response

### 2.使用msg

- Creating a Msg 
  
  **1.创建一个msg文件**
	
	```
	$ roscd beginner_tutorials
	$ mkdir msg
	$ echo "int64 num" > msg/Num.msg	#输出消息到一个新的文件
  ```
  
  - `rosmsg`可以用来查看已有的msg:
  
    - 使用：`$ rosmsg show [message type]`
  
    - 例子：`$ rosmsg show beginner_tutorials/Num `
  
      如果忘了包名：`$ rosmsg show Num`

   **2.还可以`$ rosed beginner_tutorials Num.msg`,然后在里面输入**
  
    ```
    string first_name
    string last_name
    uint8 age
    uint32 score
    ```
  
  **3.确保能将msg文件转为c++，python或其他语言**
  
  ​	**3.1**：用`rosed beginner_tutorials package.xml`打开配置文件package.xml，加入如下两行
  
  ```xml
  <!--在build time我们需要message_generation-->
  <build_depend>message_generation</build_depend>
  <!--在run time我们需要message_runtime-->
  <exec_depend>message_runtime</exec_depend>
  ```
  
  ​	**3.2**：用`rosed beginner_tutorials CMakeLists.txt`打开编译文件CMakeLists.tx,
  
  ​		(1)将message_generation依赖加入find_package。
  
  ```
  # Do not just add this to your CMakeLists.txt, modify the existing text to add message_generation before the closing parenthesis
  find_package(catkin REQUIRED COMPONENTS
     roscpp
     rospy
     std_msgs
     message_generation
  )
  ```
  
  ​		(2)将message_runtime依赖加入catkin_package。
  
  ```
  catkin_package(
    ...
    CATKIN_DEPENDS message_runtime ...
    ...)
  ```
  
  ​		(3)修改add_message_files为：
  
  ```
  /*通过手动添加.msg文件，确保CMake在我们添加了其他.msg文件后会重新配置项目*/
  add_message_files(
    FILES
    Num.msg
  )
  ```
  
  ​		(4)将generate_messages()函数投入使用，出去#
  
  ```
  generate_messages(
    DEPENDENCIES
    std_msgs
  )
  ```

### 3.使用srv

- Creating a srv

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

  - 使用`rossrv`来查看当前已有的srv文件

    - 使用：`$ rossrv show <service type>`

    - 例子:

      ```
      $ rossrv show beginner_tutorials/AddTwoInts
      ```

      不记得包名：

      ```
      $ rossrv show AddTwoInts
      ```

  **3. 确保srv文件能转为c++,python或其他语言**

  ​	**3.1**: 编辑包内的package.xml,加入如下两行（当然在创建msg时已设置）

  ```xml
  <build_depend>message_generation</build_depend><exec_depend>message_runtime</exec_depend>
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

### 4. 重新编译

添加完msg和srv后需要重新编译

```
# 需要在工作空间里使用catkin_make
$ roscd beginner_tutorials
$ cd ../..
$ catkin_make
$ cd -
```

## 五、写一个简单的Publisher 和 Subscriber

### 五(一)使用c++来写

### 五(二)使用python来写

