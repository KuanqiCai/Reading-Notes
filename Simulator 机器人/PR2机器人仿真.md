# 1. Environment

1.  Ubuntu20.04 + ros noetic

2.  install PR2 Packages from [pr2_github](https://github.com/PR2) into the ws/src folder

   - [pr2_common](https://github.com/PR2/pr2_common)
   - [pr2_simulator](https://github.com/PR2/pr2_simulator)
   - [pr2_controllers](https://github.com/PR2/pr2_controllers)

   build them together with command `catkin build`

   If anything error, go to step 3.

3. install some packages necessary for the above packages

   - [ivcon](http://wiki.ros.org/ivcon):`sudo apt-get install ros-noetic-ivcon`
   - [convex_decomposition](http://wiki.ros.org/convex_decomposition):`sudo apt-get install ros-noetic-convex-decomposition`
   - [pr2_mechanism](http://wiki.ros.org/pr2_mechanism?distro=noetic):`sudo apt-get install ros-noetic-pr2-mechanism`
   - notes:
     - If you still miss any packages **in** RosWiki, just use the command `apt-get`to install.
     - If you still miss any packages **not in** RosWiki, just install the source code from github and put them into the work space/src folder.

   

# 2. Simulation

## 2.1 Start simulation 

- Start Gazebo with an empty world

  ```
   roslaunch gazebo_ros empty_world.launch
  ```

- Spawn PR2

    ```
     roslaunch pr2_gazebo pr2.launch
    ```

