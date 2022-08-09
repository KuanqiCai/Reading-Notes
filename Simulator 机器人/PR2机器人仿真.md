# 1. Environment

The OS is : Ubuntu20.04 + ros noetic

All PR2 Packages can be found in [pr2_github](https://github.com/PR2)

## 1.1 Install Dependencies for SDformat:

**Reference**: http://sdformat.org/tutorials?tut=install

1. `sudo apt-get install ruby-dev build-essential libtinyxml-dev libboost-all-dev cmake mercurial pkg-config`
2. `sudo apt-get install libignition-math4-dev`

## 1.2 Building SDformat:

Reference: http://sdformat.org/

```shell
$ mkdir ~/sdf_source
$ cd ~/sdf_source/
$ git clone https://github.com/osrf/sdformat
$ cd sdformat
$ git checkout sdf6
$ mkdir build
$ cd build
$ cmake .. -DCMAKE_INSTALL_PREFIX=/usr
$ make -j4
$ make install
```

## 1.3 install ROS Pkgs

- [ivcon](http://wiki.ros.org/ivcon):`sudo apt-get install ros-noetic-ivcon`
- [convex_decomposition](http://wiki.ros.org/convex_decomposition):`sudo apt-get install ros-noetic-convex-decomposition`

## 1.4 install PR2

- [pr2_mechanism](http://wiki.ros.org/pr2_mechanism?distro=noetic):`sudo apt-get install ros-noetic-pr2-mechanism`
- pr2_simulator:`sudo apt-get install ros-noetic-pr2-simulator`
- pr2_common:`sudo apt-get install ros-noetic-pr2-common`
- pr2_mechanism_msgs:`sudo apt-get install ros-noetic-pr2-mechanism-msgs`

- pr2_teleop:`sudo apt-get install ros-noetic-pr2-teleop`
- PR2 Dashboard:`sudo apt-get install ros-noetic-rqt-pr2-dashboard `
- Navigation:`sudo apt-get install ros-noetic-navigation-perception ros-noetic-pr2-navigation-teleop ros-noetic-pr2-navigation-global `

- ....

note:

- For any packages **in** RosWiki, just use the command `apt-get`to install.

  The files are installed in /opt/ros/noetic/share

- For any packages **not in** RosWiki, just install the source code from github and put them into the work space/src folder.

  eg:

  ```
  $ mkdir ws_sa
  $ cd ws_sa/
  $ mkdir src
  $ cd src/
  $ git clone https://github.com/PR2/pr2_mechanism.git
  $ git clone https://github.com/PR2/pr2_common.git
  $ git clone https://github.com/PR2/pr2_mechanism_msgs.git
  $ cd ..
  $ catkin_make
  ```

  



# 2. Simulation

## 2.1 Demo 

- Start Gazebo with an empty world

  ```
   roslaunch gazebo_ros empty_world.launch
  ```

- Spawn PR2

    ```
     roslaunch pr2_gazebo pr2.launch
    ```
    
    pr2.launch:
    
    ```xml
    <launch>
    
      <!-- Startup PR2 without any mechanism controllers -->
      <include file="$(find pr2_gazebo)/pr2_no_controllers.launch" />
    
      <!-- Load and Start Default Controllers -->
      <include file="$(find pr2_controller_configuration_gazebo)/pr2_default_controllers.launch" />
    
    </launch>
    ```
    
    1. [pr2_no_controllers.launch](https://github.com/PR2/pr2_simulator/blob/kinetic-devel/pr2_gazebo/launch/pr2_no_controllers.launch)
    
    2. [pr2_default_controllers.launch](https://github.com/PR2/pr2_simulator/blob/kinetic-devel/pr2_controller_configuration_gazebo/launch/pr2_default_controllers.launch)
    
       

## 2.2 Simulator Workshop

### 2.2.1 PR2 Simulator

- In .bashrc add:

  ```
  export ROS_MASTER_URI=http://localhost:11311
  export ROBOT=sim
  ```

- run:`roslaunch pr2_gazebo pr2_empty_world.launch`

### 2.2.2 Control the Mobile Base

- run:`roslaunch pr2_teleop teleop_keyboard.launch`

### 2.2.3 Use topic to control one joint

- **Find an appropriate topic**

  `rostopic list | grep r_gripper`

  Then choose a topic,eg:"/r_gripper_controller/command"

- **Check the type of this topic's message**

  `rostopic info /r_gripper_controller/command`

  The message type is :"pr2_controllers_msgs/Pr2GripperCommand"

- **Check the msg's detail**

  `rosmsg show pr2_controllers_msgs/Pr2GripperCommand`

- **Publish a msg** 

  `rostopic pub r_gripper_controller/command pr2_controllers_msgs/Pr2GripperCommand "{position: 0.06, max_effort: 100.0}"`

### 2.2.4 PR2 Navigation

- Add Obstacle
  1. run the command like:`rosrun gazebo_ros spawn_model -urdf -file drawer.urdf -model drawer1 -x 1.0`
  2. use the buttons in Gazebo

- Run the Nav
  - `roslaunch test_controller pr2_nav.launch `
  - `roslaunch pr2_navigation_global rviz_move_base.launch`
