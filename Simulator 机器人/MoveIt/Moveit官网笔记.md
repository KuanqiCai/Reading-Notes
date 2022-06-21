# 参考资料

- [URDF](https://ros-planning.github.io/moveit_tutorials/doc/urdf_srdf/urdf_srdf_tutorial.html)
- 

# 一、使用RVIZ看Moveit

- 运行：`roslaunch panda_moveit_config demo.launch rviz_tutorial:=true`

# 二、Move Group C++ Interface

- 运行2个shell,分别执行:
  - `roslaunch panda_moveit_config demo.launch`
  - `roslaunch moveit_tutorials move_group_interface_tutorial.launch`

- 简介
  - MoveIt主要的用户接口功能通过[MoveGroup](http://docs.ros.org/indigo/api/moveit_ros_planning_interface/html/classmoveit_1_1planning__interface_1_1MoveGroup.html)类实现
  - 这个类提供简易方式去实现大部分功能，比如：设置关节或目标姿态，创建行为规划，移动机器人，在环境中增加对象或给机器人增加或减少对象。

- 实现的功能：

  - The robot moves its arm to the pose goal to its front.
  - The robot moves its arm to the joint goal at its side.
  - The robot moves its arm back to a new pose goal while maintaining the end-effector level.
  - The robot moves its arm along the desired Cartesian path (a triangle down, right, up+left).
  - A box object is added into the environment to the right of the arm.
  - The robot moves its arm to the pose goal, avoiding collision with the box.
  - The object is attached to the wrist手腕 (its color will change to purple/orange/green).
  - The object is detached from the wrist (its color will change back to green).
  - The object is removed from the environment.

- 代码：

  ```c++
  #include <moveit/move_group_interface/move_group_interface.h>
  #include <moveit/planning_scene_interface/planning_scene_interface.h>
  
  #include <moveit_msgs/DisplayRobotState.h>
  #include <moveit_msgs/DisplayTrajectory.h>
  
  #include <moveit_msgs/AttachedCollisionObject.h>
  #include <moveit_msgs/CollisionObject.h>
  
  #include <moveit_visual_tools/moveit_visual_tools.h>
  
  // The circle constant tau = 2*pi. One tau is one rotation in radians.
  const double tau = 2 * M_PI;
  
  int main(int argc, char** argv)
  {
    ros::init(argc, argv, "move_group_interface_tutorial");
    ros::NodeHandle node_handle;
  
    // ROS spinning must be running for the MoveGroupInterface to get information
    // about the robot's state. One way to do this is to start an AsyncSpinner
    // beforehand.
    // ros::AsyncSpinner创建多线程，1表示创建1个新的线程。
    // 它有start()和stop()调用, 并且在销毁时自动停止
    ros::AsyncSpinner spinner(1);
    spinner.start();
  
      
    // BEGIN_TUTORIAL
    //********************** Setup **********************
    /* MoveIt operates on sets of joints called "planning groups" and stores them in an object called the `JointModelGroup`. Throughout MoveIt the terms "planning group" and "joint model group" are used interchangeably可替换的.
    MoveIt 对称为“planning groups”的关节集合进行操作，并将它们存储在称为 JointModelGroup 的对象中。在整个 MoveIt 中，术语“planning groups”和“joint model group”可以互换使用。
    */
    static const std::string PLANNING_GROUP = "panda_arm";
  
    /* The :planning_interface:`MoveGroupInterface` class can be easily setup using just the name of the planning group you would like to control and plan for.
    创建一个MoveGroupInterface类，并提供我们想要控制的关节组的名字作为参数，这里是：PLANNING_GROUP
    */
    moveit::planning_interface::MoveGroupInterface move_group_interface(PLANNING_GROUP);
  
    // We will use the :planning_interface:`PlanningSceneInterface` class to add and remove collision objects in our "virtual world" scene
    // 使用planning_scene_interface类去处理碰撞物
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
  
    // Raw pointers are frequently used to refer to the planning group for improved performance.
    // 用指针来提高性能
    const moveit::core::JointModelGroup* joint_model_group =
        move_group_interface.getCurrentState()->getJointModelGroup(PLANNING_GROUP);
  
      
    //********************** Visualization **********************
    // The package MoveItVisualTools provides many capabilities for visualizing objects, robots,and trajectories in RViz as well as debugging tools such as step-by-step introspection of a script.
    // 包MoveItVisualTools提供了用于在RViz中可视化的工具
    namespace rvt = rviz_visual_tools;
    moveit_visual_tools::MoveItVisualTools visual_tools("panda_link0");
    visual_tools.deleteAllMarkers();
  
    // Remote control is an introspection tool that allows users to step through单步调试 a high level script via buttons and keyboard shortcuts in RViz
    visual_tools.loadRemoteControl();
  
    // RViz provides many types of markers, in this demo we will use text, cylinders圆柱, and spheres球体
    Eigen::Isometry3d text_pose = Eigen::Isometry3d::Identity();
    text_pose.translation().z() = 1.0;
    visual_tools.publishText(text_pose, "MoveGroupInterface Demo", rvt::WHITE, rvt::XLARGE);
  
    // Batch publishing is used to reduce the number of messages being sent to RViz for large visualizations
    visual_tools.trigger();
  
  
    //********************** Getting Basic Information **********************
    // We can print the name of the reference frame for this robot.
    ROS_INFO_NAMED("tutorial", "Planning frame: %s", move_group_interface.getPlanningFrame().c_str());
  
    // We can also print the name of the end-effector link for this group.
    ROS_INFO_NAMED("tutorial", "End effector link: %s", move_group_interface.getEndEffectorLink().c_str());
  
    // We can get a list of all the groups in the robot:
    ROS_INFO_NAMED("tutorial", "Available Planning Groups:");
    std::copy(move_group_interface.getJointModelGroupNames().begin(),
              move_group_interface.getJointModelGroupNames().end(), std::ostream_iterator<std::string>(std::cout, ", "));
  
  
    //********************** Start the demo **********************
    visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to start the demo");
  
     
    //********************** Planning to a Pose goal **********************
    // We can plan a motion for this group to a desired pose for the end-effector.
    // 设置一个想要末端执行器到达的姿势
    geometry_msgs::Pose target_pose1;
    target_pose1.orientation.w = 1.0;
    target_pose1.position.x = 0.28;
    target_pose1.position.y = -0.2;
    target_pose1.position.z = 0.5;
    move_group_interface.setPoseTarget(target_pose1);
  
    // Now, we call the planner to compute the plan and visualize it. 
    // Note that we are just planning, not asking move_group_interface to actually move the robot.
    // 定义一个plan规划
    moveit::planning_interface::MoveGroupInterface::Plan my_plan;
    // 计算规划，并用布尔型变量标记运动规划是否成功
    bool success = (move_group_interface.plan(my_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
  
    ROS_INFO_NAMED("tutorial", "Visualizing plan 1 (pose goal) %s", success ? "" : "FAILED");
  
  
    //********************** Visualizing plans **********************
    // We can also visualize the plan as a line with markers in RViz.
    ROS_INFO_NAMED("tutorial", "Visualizing plan 1 as trajectory line");
    visual_tools.publishAxisLabeled(target_pose1, "pose1");
    visual_tools.publishText(text_pose, "Pose Goal", rvt::WHITE, rvt::XLARGE);
    visual_tools.publishTrajectoryLine(my_plan.trajectory_, joint_model_group);
    visual_tools.trigger();
    visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to continue the demo");
  
  
    //********************** Moving to a pose goal **********************
    // Finally, to execute the trajectory stored in my_plan, you could use the following method call:
    // Note that this can lead to problems if the robot moved in the meanwhile.
    // move_group_interface.execute(my_plan);
      
    // If you do not want to inspect检查 the planned trajectory,the following is a more robust combination of the two-step plan+execute pattern shown above and should be preferred. Note that the pose goal we had set earlier is still active,so the robot will try to move to that goal.
    // move_group_interface.move();
    /* 有两种选择让机器人移动到想要到的地方：
    	1.一种是先plan()再execute()
    	2.另一种是直接用move()
    	  move()是一个阻塞函数，需要一个控制器是激活，执行后报告成功的轨迹
    */ 
  
     
    //********************** Planning to a joint-space goal **********************
    // Let's set a joint space goal关节空间内的坐标 and move towards it.  This will replace the pose target we set above.
    // 第一步：得到机器人当前所有的状态（位置，速度，加速度）
    // We'll create an pointer that references the current robot's state.
    // RobotState is the object that contains all the current position/velocity/acceleration data.
    moveit::core::RobotStatePtr current_state = move_group_interface.getCurrentState();
    // 第二步：从上面的状态信息中得到各个关节的位置
    // Next get the current set of joint values for the group.
    std::vector<double> joint_group_positions;
    current_state->copyJointGroupPositions(joint_model_group, joint_group_positions);
    // 第三步：设置新的关节位置
    // Now, let's modify one of the joints, plan to the new joint space goal and visualize the plan.
    joint_group_positions[0] = -tau / 6;  // -1/6 turn in radians
    move_group_interface.setJointValueTarget(joint_group_positions); 
    // 第四步：设置机器人这次移动的允许最大速度（可执行最大速度的百分比）
    // We lower the allowed maximum velocity and acceleration to 5% of their maximum.
    // The default values are 10% (0.1).
    // Set your preferred defaults in the joint_limits.yaml file of your robot's moveit_config or set explicit factors in your code if you need your robot to move faster.
    move_group_interface.setMaxVelocityScalingFactor(0.05);
    move_group_interface.setMaxAccelerationScalingFactor(0.05); 
    // 第五步：规划路径，并判断是否成功
    success = (move_group_interface.plan(my_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
    ROS_INFO_NAMED("tutorial", "Visualizing plan 2 (joint space goal) %s", success ? "" : "FAILED"); 
    // 第六步：可视化这个路径（这里没有执行）
    // Visualize the plan in RViz
    visual_tools.deleteAllMarkers();
    visual_tools.publishText(text_pose, "Joint Space Goal", rvt::WHITE, rvt::XLARGE);
    visual_tools.publishTrajectoryLine(my_plan.trajectory_, joint_model_group);
    visual_tools.trigger();
    visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to continue the demo");
  
      
    //********************** Planning with Path Constraints **********************
    // 有路径约束时的规划
    // Path constraints can easily be specified for a link on the robot.
    // Let's specify a path constraint and a pose goal for our group.
    // First define the path constraint.
    moveit_msgs::OrientationConstraint ocm;
    ocm.link_name = "panda_link7";
    ocm.header.frame_id = "panda_link0";
    ocm.orientation.w = 1.0;
    ocm.absolute_x_axis_tolerance = 0.1;
    ocm.absolute_y_axis_tolerance = 0.1;
    ocm.absolute_z_axis_tolerance = 0.1;
    ocm.weight = 1.0;
    // Now, set it as the path constraint for the group.
    moveit_msgs::Constraints test_constraints;
    test_constraints.orientation_constraints.push_back(ocm);
    move_group_interface.setPathConstraints(test_constraints);
  
  
    //********************** Enforce Planning in Joint Space **********************
    /*Depending on the planning problem MoveIt chooses between ``joint space`` and ``cartesian space`` for problem representation.
    Setting the group parameter ``enforce_joint_model_state_space:true`` in the ompl_planning.yaml file enforces the use of ``joint space`` for all plans.
    By default planning requests with orientation path constraints are sampled in ``cartesian space`` so that invoking IK serves as a generative sampler生成采样器。默认情况下，具有方向路径约束的规划请求在笛卡尔空间中进行采样。
    By enforcing ``joint space`` the planning process will use rejection sampling拒绝抽样 to find valid requests. Please note that this might increase planning time considerably相当的.
    We will reuse the old goal that we had and plan to it.
    Note that this will only work if the current state already satisfies the path constraints. So we need to set the start state to a new pose.*/
    // 第一步：设置初始位姿到符合path constraints的位置
    moveit::core::RobotState start_state(*move_group_interface.getCurrentState());
    geometry_msgs::Pose start_pose2;
    start_pose2.orientation.w = 1.0;
    start_pose2.position.x = 0.55;
    start_pose2.position.y = -0.05;
    start_pose2.position.z = 0.8;
    start_state.setFromIK(joint_model_group, start_pose2);
    move_group_interface.setStartState(start_state);
    // Now we will plan to the earlier pose target from the new start state that we have just created.
    move_group_interface.setPoseTarget(target_pose1);
    // 第二步：因为计算约束很慢，所以增加计算时间
    // Planning with constraints can be slow because every sample must call an inverse kinematics solver.
    // Lets increase the planning time from the default 5 seconds to be sure the planner has enough time to succeed.
    move_group_interface.setPlanningTime(10.0);
    // 第三步：计算运动规划，并检测是否成功
    success = (move_group_interface.plan(my_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
    ROS_INFO_NAMED("tutorial", "Visualizing plan 3 (constraints) %s", success ? "" : "FAILED");
    // 第四步：可视化
    // Visualize the plan in RViz
    visual_tools.deleteAllMarkers();
    visual_tools.publishAxisLabeled(start_pose2, "start");
    visual_tools.publishAxisLabeled(target_pose1, "goal");
    visual_tools.publishText(text_pose, "Constrained Goal", rvt::WHITE, rvt::XLARGE);
    visual_tools.publishTrajectoryLine(my_plan.trajectory_, joint_model_group);
    visual_tools.trigger();
    visual_tools.prompt("next step");
    // 第五步：计算完后要清除约束
    // When done with the path constraint be sure to clear it.
    move_group_interface.clearPathConstraints();
  
    
    //********************** Cartesian Paths **********************
    // 机械臂在笛卡尔空间的运动只能走点到点的直线运动，通过将位姿点加入waypoints中，机械臂会一次按照waypoints中的唯一依次沿直线运动到下一个点。
    // You can plan a Cartesian path directly by specifying a list of waypoints for the end-effector to go through. Note that we are starting from the new start state above.  The initial pose (start state) does not need to be added to the waypoint list but adding it can help with visualizations
    // 第一步：设置路径点
    std::vector<geometry_msgs::Pose> waypoints;
    waypoints.push_back(start_pose2);
    geometry_msgs::Pose target_pose3 = start_pose2;
    target_pose3.position.z -= 0.2;
    waypoints.push_back(target_pose3);  // down
    target_pose3.position.y -= 0.2;
    waypoints.push_back(target_pose3);  // right
    target_pose3.position.z += 0.2;
    target_pose3.position.y += 0.2;
    target_pose3.position.x -= 0.2;
    waypoints.push_back(target_pose3);  // up and left
    // 第二步：计算路径
    // We want the Cartesian path to be interpolated at a resolution分辨率 of 1 cm which is why we will specify 0.01 as the max step in Cartesian translation.  We will specify the jump threshold阈值 as 0.0, effectively disabling it.
    // Warning - disabling the jump threshold while operating real hardware can cause large unpredictable motions of redundant joints and could be a safety issue
    moveit_msgs::RobotTrajectory trajectory;
    const double jump_threshold = 0.0;
    const double eef_step = 0.01;
    double fraction = move_group_interface.computeCartesianPath(waypoints, eef_step, jump_threshold, trajectory);
    ROS_INFO_NAMED("tutorial", "Visualizing plan 4 (Cartesian path) (%.2f%% achieved)", fraction * 100.0);
    // 第三步：可视化路径
    // Visualize the plan in RViz
    visual_tools.deleteAllMarkers();
    visual_tools.publishText(text_pose, "Cartesian Path", rvt::WHITE, rvt::XLARGE);
    visual_tools.publishPath(waypoints, rvt::LIME_GREEN, rvt::SMALL);
    for (std::size_t i = 0; i < waypoints.size(); ++i)
      visual_tools.publishAxisLabeled(waypoints[i], "pt" + std::to_string(i), rvt::SMALL);
    visual_tools.trigger();
    visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to continue the demo");
    // Cartesian motions should often be slow, e.g. when approaching objects. The speed of cartesian plans cannot currently be set through the maxVelocityScalingFactor, but requires you to time the trajectory manually, as described `here <https://groups.google.com/forum/#!topic/moveit-users/MOoFxy2exT4>`_.
    // 第四步：执行路径
    // You can execute a trajectory like this.
    move_group_interface.execute(trajectory);
  
  
    //********************** Adding objects to the environment **********************
    // First let's plan to another simple goal with no objects in the way.
    // 第一步：没有物体的简单运动规划
    move_group_interface.setStartState(*move_group_interface.getCurrentState());
    geometry_msgs::Pose another_pose;
    another_pose.orientation.x = 1.0;
    another_pose.position.x = 0.7;
    another_pose.position.y = 0.0;
    another_pose.position.z = 0.59;
    move_group_interface.setPoseTarget(another_pose);
  
    success = (move_group_interface.plan(my_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
    ROS_INFO_NAMED("tutorial", "Visualizing plan 5 (with no obstacles) %s", success ? "" : "FAILED");
  
    visual_tools.deleteAllMarkers();
    visual_tools.publishText(text_pose, "Clear Goal", rvt::WHITE, rvt::XLARGE);
    visual_tools.publishTrajectoryLine(my_plan.trajectory_, joint_model_group);
    visual_tools.trigger();
    visual_tools.prompt("next step");
    // 第二步：添加一个要避免碰撞的物体
    // Now let's define a collision object ROS message for the robot to avoid.
    moveit_msgs::CollisionObject collision_object;
    collision_object.header.frame_id = move_group_interface.getPlanningFrame();
    // The id of the object is used to identify it.
    collision_object.id = "box1";
    // Define a box to add to the world.
    shape_msgs::SolidPrimitive primitive;
    primitive.type = primitive.BOX;
    primitive.dimensions.resize(3);
    primitive.dimensions[primitive.BOX_X] = 0.1;
    primitive.dimensions[primitive.BOX_Y] = 1.5;
    primitive.dimensions[primitive.BOX_Z] = 0.5;
    // Define a pose for the box (specified relative to frame_id)
    geometry_msgs::Pose box_pose;
    box_pose.orientation.w = 1.0;
    box_pose.position.x = 0.5;
    box_pose.position.y = 0.0;
    box_pose.position.z = 0.25;
    // 将上面定义好的盒子信息传入moveit_msgs
    collision_object.primitives.push_back(primitive);
    collision_object.primitive_poses.push_back(box_pose);
    collision_object.operation = collision_object.ADD;
    // 如果有多个，就搞个vector
    std::vector<moveit_msgs::CollisionObject> collision_objects;
    collision_objects.push_back(collision_object);
    // 第三步：将定义好的障碍物放入我们的模拟中
    // Now, let's add the collision object into the world
    // (using a vector that could contain additional objects)
    ROS_INFO_NAMED("tutorial", "Add an object into the world");
    planning_scene_interface.addCollisionObjects(collision_objects);
    // 第四步：在rviz中展现障碍物
    // Show text in RViz of status and wait for MoveGroup to receive and process the collision object message
    visual_tools.publishText(text_pose, "Add object", rvt::WHITE, rvt::XLARGE);
    visual_tools.trigger();
    visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to once the collision object appears in RViz");
    // 第五步：规划一个可以避障的运动规划
    // Now when we plan a trajectory it will avoid the obstacle
    success = (move_group_interface.plan(my_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
    ROS_INFO_NAMED("tutorial", "Visualizing plan 6 (pose goal move around cuboid) %s", success ? "" : "FAILED");
    visual_tools.publishText(text_pose, "Obstacle Goal", rvt::WHITE, rvt::XLARGE);
    visual_tools.publishTrajectoryLine(my_plan.trajectory_, joint_model_group);
    visual_tools.trigger();
    visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window once the plan is complete");
    // 第六步：带着物体避障的运动规划-定义物体
    // You can attach objects to the robot, so that it moves with the robot geometry.
    // This simulates picking up the object for the purpose of manipulating it.
    // The motion planning should avoid collisions between the two objects as well.
    moveit_msgs::CollisionObject object_to_attach;
    object_to_attach.id = "cylinder1";
    // 物体的几何特征
    shape_msgs::SolidPrimitive cylinder_primitive;
    cylinder_primitive.type = primitive.CYLINDER;
    cylinder_primitive.dimensions.resize(2);
    cylinder_primitive.dimensions[primitive.CYLINDER_HEIGHT] = 0.20;
    cylinder_primitive.dimensions[primitive.CYLINDER_RADIUS] = 0.04;
    // 物体的位置：机器人的抓手那
    // We define the frame/pose for this cylinder so that it appears in the gripper
    object_to_attach.header.frame_id = move_group_interface.getEndEffectorLink();
    geometry_msgs::Pose grab_pose;
    grab_pose.orientation.w = 1.0;
    grab_pose.position.z = 0.2;
    // 第七步：将物体信息放入我们的模拟中
    // First, we add the object to the world (without using a vector)
    object_to_attach.primitives.push_back(cylinder_primitive);
    object_to_attach.primitive_poses.push_back(grab_pose);
    object_to_attach.operation = object_to_attach.ADD;
    planning_scene_interface.applyCollisionObject(object_to_attach);
    // 第八步：选择哪个机器人抓哪个物体
    // Then, we "attach" the object to the robot. It uses the frame_id to determine which robot link it is attached to.
    // You could also use applyAttachedCollisionObject to attach an object to the robot directly.
    ROS_INFO_NAMED("tutorial", "Attach the object to the robot");
    move_group_interface.attachObject(object_to_attach.id, "panda_hand");
    // 第九步：可视化设置
    visual_tools.publishText(text_pose, "Object attached to robot", rvt::WHITE, rvt::XLARGE);
    visual_tools.trigger();
    /* Wait for MoveGroup to receive and process the attached collision object message */
    visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window once the new object is attached to the robot");
    // 第十步：计算路径规划并检查是否成功
    // Replan, but now with the object in hand.
    move_group_interface.setStartStateToCurrentState();
    success = (move_group_interface.plan(my_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
    ROS_INFO_NAMED("tutorial", "Visualizing plan 7 (move around cuboid with cylinder) %s", success ? "" : "FAILED");
    visual_tools.publishTrajectoryLine(my_plan.trajectory_, joint_model_group);
    visual_tools.trigger();
    visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window once the plan is complete");
  
    
    //********************** Detaching and Removing Objects **********************
    // Now, let's detach the cylinder from the robot's gripper.
    ROS_INFO_NAMED("tutorial", "Detach the object from the robot");
    move_group_interface.detachObject(object_to_attach.id);
    // Show text in RViz of status
    visual_tools.deleteAllMarkers();
    visual_tools.publishText(text_pose, "Object detached from robot", rvt::WHITE, rvt::XLARGE);
    visual_tools.trigger();
    /* Wait for MoveGroup to receive and process the attached collision object message */
    visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window once the new object is detached from the robot");
    // Now, let's remove the objects from the world.
    ROS_INFO_NAMED("tutorial", "Remove the objects from the world");
    std::vector<std::string> object_ids;
    object_ids.push_back(collision_object.id);
    object_ids.push_back(object_to_attach.id);
    planning_scene_interface.removeCollisionObjects(object_ids);
    // Show text in RViz of status
    visual_tools.publishText(text_pose, "Objects removed", rvt::WHITE, rvt::XLARGE);
    visual_tools.trigger();
    /* Wait for MoveGroup to receive and process the attached collision object message */
    visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to once the collision object disappears");
    // END_TUTORIAL
  
    ros::shutdown();
    return 0;
  }
  ```
  

# 三、MoveIt Commander

- 简介：

  The [moveit_commander](http://wiki.ros.org/moveit_commander) Python package offers wrappers封装 for the functionality provided in MoveIt. Simple interfaces are available for motion planning, computation of Cartesian paths, and pick and place.

- 两个shell运行：

  - `roslaunch panda_moveit_config demo.launch`
  - `rosrun moveit_commander moveit_commander_cmdline.py`

- 然后可在上面第二个shell中控制机器人

  - `use panda_arm`

    选择我们想要控制的的joint group的名字。 This will connect you to a running instance of the move_group node

  - `current`

    显示当前机器人所有的状态

  - `rec c`

    将当前joint values保存到c

  - 让机器人到我们想到的地方

    - 直接跑

      ```
      goal = c
      goal[0] = 0.2
      go goal
      ```

    - 先规划再执行

      ```
      goal[0] = 0.2
      goal[1] = 0.2
      plan goal
      execute
      ```

# 四、RobotModel and RobotState Classes

- 简介

  The [RobotModel](http://docs.ros.org/noetic/api/moveit_core/html/cpp/classmoveit_1_1core_1_1RobotModel.html) and [RobotState](http://docs.ros.org/noetic/api/moveit_core/html/cpp/classmoveit_1_1core_1_1RobotState.html) classes are the core classes that give you access to a robot’s kinematics.用于计算运动学

  - RobotModel:
    - RobotModel contains the relationships between all links and joints including their joint limit properties as loaded from the URDF.
    - The RobotModel also separates the robot’s links and joints into planning groups defined in the SRDF
  - RobotState:
    - RobotState contains information about the robot at a certain point in time, storing vectors of joint positions and optionally velocities and accelerations.
    - This information can be used to obtain kinematic information about the robot that depends on its current state, such as the Jacobian of an end effector. 
    - RobotState also contains helper functions for setting the arm location based on the end effector location (Cartesian pose) and for computing Cartesian trajectories.

- 运行：

  `roslaunch moveit_tutorials robot_model_and_robot_state_tutorial.launch`

  - 计算结果在shell中展示

- 代码

  ```c++
  #include <ros/ros.h>
  
  // MoveIt
  #include <moveit/robot_model_loader/robot_model_loader.h>
  #include <moveit/robot_model/robot_model.h>
  #include <moveit/robot_state/robot_state.h>
  
  int main(int argc, char** argv)
  {
    ros::init(argc, argv, "robot_model_and_robot_state_tutorial");
    ros::AsyncSpinner spinner(1);
    spinner.start();
  
    // BEGIN_TUTORIAL
    // Start
    // ^^^^^
    // Setting up to start using the RobotModel class is very easy. In
    // general, you will find that most higher-level components will
    // return a shared pointer to the RobotModel. You should always use
    // that when possible. In this example, we will start with such a
    // shared pointer and discuss only the basic API. You can have a
    // look at the actual code API for these classes to get more
    // information about how to use more features provided by these
    // classes.
    //
    // We will start by instantiating a
    // `RobotModelLoader`_
    // object, which will look up
    // the robot description on the ROS parameter server and construct a
    // :moveit_core:`RobotModel` for us to use.
    //
    // .. _RobotModelLoader:
    //     http://docs.ros.org/noetic/api/moveit_ros_planning/html/classrobot__model__loader_1_1RobotModelLoader.html
    robot_model_loader::RobotModelLoader robot_model_loader("robot_description");
    const moveit::core::RobotModelPtr& kinematic_model = robot_model_loader.getModel();
    ROS_INFO("Model frame: %s", kinematic_model->getModelFrame().c_str());
  
    // Using the :moveit_core:`RobotModel`, we can construct a
    // :moveit_core:`RobotState` that maintains the configuration
    // of the robot. We will set all joints in the state to their
    // default values. We can then get a
    // :moveit_core:`JointModelGroup`, which represents the robot
    // model for a particular group, e.g. the "panda_arm" of the Panda
    // robot.
    moveit::core::RobotStatePtr kinematic_state(new moveit::core::RobotState(kinematic_model));
    kinematic_state->setToDefaultValues();
    const moveit::core::JointModelGroup* joint_model_group = kinematic_model->getJointModelGroup("panda_arm");
  
    const std::vector<std::string>& joint_names = joint_model_group->getVariableNames();
  
    // Get Joint Values
    // ^^^^^^^^^^^^^^^^
    // We can retrieve the current set of joint values stored in the state for the Panda arm.
    std::vector<double> joint_values;
    kinematic_state->copyJointGroupPositions(joint_model_group, joint_values);
    for (std::size_t i = 0; i < joint_names.size(); ++i)
    {
      ROS_INFO("Joint %s: %f", joint_names[i].c_str(), joint_values[i]);
    }
  
    // Joint Limits
    // ^^^^^^^^^^^^
    // setJointGroupPositions() does not enforce joint limits by itself, but a call to enforceBounds() will do it.
    /* Set one joint in the Panda arm outside its joint limit */
    joint_values[0] = 5.57;
    kinematic_state->setJointGroupPositions(joint_model_group, joint_values);
  
    /* Check whether any joint is outside its joint limits */
    ROS_INFO_STREAM("Current state is " << (kinematic_state->satisfiesBounds() ? "valid" : "not valid"));
  
    /* Enforce the joint limits for this state and check again*/
    kinematic_state->enforceBounds();
    ROS_INFO_STREAM("Current state is " << (kinematic_state->satisfiesBounds() ? "valid" : "not valid"));
  
    // Forward Kinematics
    // ^^^^^^^^^^^^^^^^^^
    // Now, we can compute forward kinematics for a set of random joint
    // values. Note that we would like to find the pose of the
    // "panda_link8" which is the most distal link in the
    // "panda_arm" group of the robot.
    kinematic_state->setToRandomPositions(joint_model_group);
    const Eigen::Isometry3d& end_effector_state = kinematic_state->getGlobalLinkTransform("panda_link8");
  
    /* Print end-effector pose. Remember that this is in the model frame */
    ROS_INFO_STREAM("Translation: \n" << end_effector_state.translation() << "\n");
    ROS_INFO_STREAM("Rotation: \n" << end_effector_state.rotation() << "\n");
  
    // Inverse Kinematics
    // ^^^^^^^^^^^^^^^^^^
    // We can now solve inverse kinematics (IK) for the Panda robot.
    // To solve IK, we will need the following:
    //
    //  * The desired pose of the end-effector (by default, this is the last link in the "panda_arm" chain):
    //    end_effector_state that we computed in the step above.
    //  * The timeout: 0.1 s
    double timeout = 0.1;
    bool found_ik = kinematic_state->setFromIK(joint_model_group, end_effector_state, timeout);
  
    // Now, we can print out the IK solution (if found):
    if (found_ik)
    {
      kinematic_state->copyJointGroupPositions(joint_model_group, joint_values);
      for (std::size_t i = 0; i < joint_names.size(); ++i)
      {
        ROS_INFO("Joint %s: %f", joint_names[i].c_str(), joint_values[i]);
      }
    }
    else
    {
      ROS_INFO("Did not find IK solution");
    }
  
    // Get the Jacobian
    // ^^^^^^^^^^^^^^^^
    // We can also get the Jacobian from the :moveit_core:`RobotState`.
    Eigen::Vector3d reference_point_position(0.0, 0.0, 0.0);
    Eigen::MatrixXd jacobian;
    kinematic_state->getJacobian(joint_model_group,
                                 kinematic_state->getLinkModel(joint_model_group->getLinkModelNames().back()),
                                 reference_point_position, jacobian);
    ROS_INFO_STREAM("Jacobian: \n" << jacobian << "\n");
    // END_TUTORIAL
  
    ros::shutdown();
    return 0;
  }
  ```

## Launch File

```xml
<launch>
  <include file="$(find panda_moveit_config)/launch/planning_context.launch">
    <arg name="load_robot_description" value="true"/>
  </include>

  <node name="robot_model_and_robot_state_tutorial"
        pkg="moveit_tutorials"
        type="robot_model_and_robot_state_tutorial"
        respawn="false" output="screen">
    <rosparam command="load"
              file="$(find panda_moveit_config)/config/kinematics.yaml"/>
  </node>
</launch>
```

## Debugging the Robot State

`rosrun moveit_ros_planning moveit_print_planning_model_info`