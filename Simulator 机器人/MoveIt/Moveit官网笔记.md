坐标轴：红绿蓝-》xyz

# MovIt工程架构

![](https://github.com/Fernweh-yang/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/ROS/moveit%20frame.png?raw=true)

MoveIt 的核心节点（node）为**move_group**，外围的几个部分分别为：**ROS Param Server**， **Robot Controllers**， **Robot 3D Sensors**， **User Interface**， **Robot Sensors**

- **ROS Param Server**：这部分载入的是用户定义的模型文件（xacro或urdf）和一些配置文件。（重要）
- **Robot Controllers**： 这部分可以看做是和真正的机器人部分（硬件控制接口）打交道的部分，即运动规划的数据由此发给机器人驱动部分，后续会详细讲解。（重要）
- **Robot 3D Sensors**： 这部分作用是载入RGB-D相机或激光雷达等获得的点云数据用于机械手的抓取或避障等。
- **User Interface**：     这部分是用户接口，MoveIt提供一系列的API供用户完成自定义的功能，这里主要。（重要）
- **Robot Sensors**：    这部分是接收机械臂的传感器数据，然后预估出机器人的状态并发布。

# 各种用到的库/类汇总

- [URDF](https://ros-planning.github.io/moveit_tutorials/doc/urdf_srdf/urdf_srdf_tutorial.html)
- [URDF Examples](https://wiki.ros.org/urdf/Examples)
- [MoveGroup](http://docs.ros.org/indigo/api/moveit_ros_planning_interface/html/classmoveit_1_1planning__interface_1_1MoveGroup.html)
- [moveit_commander](http://wiki.ros.org/moveit_commander) 
- [RobotModel](http://docs.ros.org/noetic/api/moveit_core/html/cpp/classmoveit_1_1core_1_1RobotModel.html) 
- [RobotState](http://docs.ros.org/noetic/api/moveit_core/html/cpp/classmoveit_1_1core_1_1RobotState.html)
- [PlanningScene](http://docs.ros.org/noetic/api/moveit_core/html/cpp/classplanning__scene_1_1PlanningScene.html) 
- [PlanningSceneMonitor](http://docs.ros.org/noetic/api/moveit_ros_planning/html/classplanning__scene__monitor_1_1PlanningSceneMonitor.html)
- [PlanningContext](http://docs.ros.org/en/indigo/api/moveit_core/html/classplanning__interface_1_1PlanningContext.html#af3d95dd741609c58847bd312b8b033e0)
- [Planning_interface](http://docs.ros.org/en/indigo/api/moveit_core/html/namespaceplanning__interface.html)
- 碰撞

  - [CollisionRequest](http://docs.ros.org/noetic/api/moveit_core/html/cpp/structcollision__detection_1_1CollisionRequest.html)：碰撞检测请求
  - [CollisionResult](http://docs.ros.org/noetic/api/moveit_core/html/cpp/structcollision__detection_1_1CollisionResult.html) ：碰撞检测结果
  - [AllowedCollisionMatrix](http://docs.ros.org/noetic/api/moveit_core/html/cpp/classcollision__detection_1_1AllowedCollisionMatrix.html) (ACM) 
- 约束，都来自 [KinematicConstrain](http://docs.ros.org/noetic/api/moveit_core/html/cpp/classkinematic__constraints_1_1KinematicConstraint.html)类：

  - [JointConstraint](http://docs.ros.org/noetic/api/moveit_core/html/cpp/classkinematic__constraints_1_1JointConstraint.html)
  - [PositionConstraint](http://docs.ros.org/noetic/api/moveit_core/html/cpp/classkinematic__constraints_1_1PositionConstraint.html)
  - [OrientationConstraint](http://docs.ros.org/noetic/api/moveit_core/html/cpp/classkinematic__constraints_1_1OrientationConstraint.html) 
  - [VisibilityConstraint](http://docs.ros.org/noetic/api/moveit_core/html/cpp/classkinematic__constraints_1_1VisibilityConstraint.html) 
  - 用于设置这些constraint的函数定义在 [utils.h](http://docs.ros.org/noetic/api/moveit_core/html/cpp/utils_8h.html) file
- [moveit_visual_tools](http://docs.ros.org/en/kinetic/api/moveit_visual_tools/html/classmoveit__visual__tools_1_1MoveItVisualTools.html)

  - [rviz_visual_tools](http://docs.ros.org/en/kinetic/api/rviz_visual_tools/html/classrviz__visual__tools_1_1RvizVisualTools.html)
- [moveit_msgs](http://docs.ros.org/en/lunar/api/moveit_msgs/html/index-msg.html)
- [ROS pluginlib](http://wiki.ros.org/pluginlib)
- [InteractiveRobot](http://docs.ros.org/en/groovy/api/pr2_moveit_tutorials/html/classInteractiveRobot.html)

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

## 代码：

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

- 然后可在上面第二个shell中控制机器

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

  相关类的方法都可以在以上网站查到

  - RobotModel:
    - RobotModel 负责从 URDF、SRDF中抓取该手臂的几何参数，比如:杆件参数、关节参数..等等，
    - RobotModel contains the relationships between all links and joints including their joint limit properties as loaded from the URDF.
    - The RobotModel also separates the robot’s links and joints into planning groups defined in the SRDF
    - [RobotModelLoader类](http://docs.ros.org/noetic/api/moveit_ros_planning/html/classrobot__model__loader_1_1RobotModelLoader.html)：获取模型。会从ros parameter server中读取机器人描述，并构建一个 [RobotModel](http://docs.ros.org/noetic/api/moveit_core/html/cpp/classmoveit_1_1core_1_1RobotModel.html) 来使用
      - 可以使用`rosparam list`、`rosparam get /robot_description`来查看内容
  - RobotState:
    - RobotState 则是可以进行运动学上的操作。
    - RobotState contains information about the robot at a certain point in time, storing vectors of joint positions and optionally velocities and accelerations.
    - This information can be used to obtain kinematic information about the robot that depends on its current state, such as the Jacobian of an end effector. 
    - RobotState also contains helper functions for setting the arm location based on the end effector location (Cartesian pose) and for computing Cartesian trajectories.

- 运行：

  `roslaunch moveit_tutorials robot_model_and_robot_state_tutorial.launch`

  - 计算结果在shell中展示

## 代码

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


//********************** Start **********************
// instantiate一个RobotModelLoader类来从urdf中读取模型。
// 运行`rosparam list`、`rosparam get /robot_description`可以发现，读取的是panda_arm.urdf.xacro
robot_model_loader::RobotModelLoader robot_model_loader("robot_description");
// 获得机器人模型RobotModel。其中RobotModelPtr的Ptr指的是指针类型。直接用RobotModel也可以（即不用指针）。
const moveit::core::RobotModelPtr& kinematic_model = robot_model_loader.getModel();
// 获得现在用的坐标系，运行后会发现是:world
ROS_INFO("Model frame: %s", kinematic_model->getModelFrame().c_str());

// 得到上面Robotmodel模型 kinematic_model的配置configuration即kinematic_state.
// RobotStatePtr指明一个指针类型的实例，new运算符是动态分配内存。
moveit::core::RobotStatePtr kinematic_state(new moveit::core::RobotState(kinematic_model));
// 将所有joints设置到默认位置0，如果0不在边界内，就设置为max+min的一半。
kinematic_state->setToDefaultValues();
// 得到一个特定关节组joint group的模型（这里是panda_arm组），即我们要控制哪些joint
const moveit::core::JointModelGroup* joint_model_group = kinematic_model->getJointModelGroup("panda_arm");
// 得到构建这个state的所有变量的名字
const std::vector<std::string>& joint_names = joint_model_group->getVariableNames();


//********************** Get Joint Values **********************
// We can retrieve得到 the current set of joint values stored in the state for the Panda arm.
std::vector<double> joint_values;
// 将指定组的joint的值复制给joint_values数组
kinematic_state->copyJointGroupPositions(joint_model_group, joint_values);
for (std::size_t i = 0; i < joint_names.size(); ++i)
{
  ROS_INFO("Joint %s: %f", joint_names[i].c_str(), joint_values[i]);
}


//********************** Joint Limits **********************
joint_values[0] = 5.57;
// setJointGroupPositions(),无论如何都改变joint的值为新值。
kinematic_state->setJointGroupPositions(joint_model_group, joint_values);
ROS_INFO_STREAM("Current state is " << (kinematic_state->satisfiesBounds() ? "valid" : "not valid"));
// enforceBounds()，同样用来改变joint的值，但如果新值不在边界内，它会自动设置为最大值/最小值
kinematic_state->enforceBounds();
ROS_INFO_STREAM("Current state is " << (kinematic_state->satisfiesBounds() ? "valid" : "not valid"));


//********************** Forward Kinematics **********************
// 计算末端link应该到的pose
// 将每个joint的值设置为边界内的任意值
kinematic_state->setToRandomPositions(joint_model_group);
// panda_link8"是机器人pada_arm最远端most distal的link
// getGlobalLinkTransform()获得link的位姿pose
const Eigen::Isometry3d& end_effector_state = kinematic_state->getGlobalLinkTransform("panda_link8");
// Print end-effector pose. Remember that this is in the model frame 
// 输出末端执行器的位姿（位移+旋转）
ROS_INFO_STREAM("Translation: \n" << end_effector_state.translation() << "\n");
ROS_INFO_STREAM("Rotation: \n" << end_effector_state.rotation() << "\n");


//**********************Inverse Kinematics **********************
// 每一次迭代/尝试attempt的时间
double timeout = 0.1;
// 计算逆动力学 
// setFromIK()有14个重载，第一个参数是控制的关节组，第二个参数是最后一个link应该到的pose
bool found_ik = kinematic_state->setFromIK(joint_model_group, end_effector_state, timeout);
// 如果成功找到了逆动力学解，会返回1
if (found_ik)
{
  // 看看逆解后每个关节joint的值
  kinematic_state->copyJointGroupPositions(joint_model_group, joint_values);
  for (std::size_t i = 0; i < joint_names.size(); ++i)
  {
    ROS_INFO("Joint %s: %f", joint_names[i].c_str(), joint_values[i]);
  }
}
// 如果没找到逆动力学解，会返回0
else
{
  ROS_INFO("Did not find IK solution");
}


//********************** Get the Jacobian **********************
// We can also get the Jacobian from the :moveit_core:`RobotState`.
Eigen::Vector3d reference_point_position(0.0, 0.0, 0.0);
Eigen::MatrixXd jacobian;
// getJacobian()参照with reference to一个link的一个点，来计算一个关节组(参数1)的jocobian
// Jacobian用于计算机器人中运动学的微分问题
kinematic_state->getJacobian(joint_model_group,
                             kinematic_state->getLinkModel(joint_model_group->getLinkModelNames().back()),
                             reference_point_position, jacobian);
ROS_INFO_STREAM("Jacobian: \n" << jacobian << "\n");

    
//********************** END_TUTORIAL **********************
ros::shutdown();
return 0;
}
```

## Launch Fil

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

# 五、Planning Scene

## 5.1 Planning Scene

- The [PlanningScene](http://docs.ros.org/noetic/api/moveit_core/html/cpp/classplanning__scene_1_1PlanningScene.html) class provides the main interface that you will use for collision checking and constraint checking. 

  用于检查碰撞和约束

  - 但用[PlanningScene](http://docs.ros.org/noetic/api/moveit_core/html/cpp/classplanning__scene_1_1PlanningScene.html) class 来实例化不是推荐的方式，更推荐的使用5.1的[PlanningSceneMonitor](http://docs.ros.org/noetic/api/moveit_ros_planning/html/classplanning__scene__monitor_1_1PlanningSceneMonitor.html)类来实例化。
  - 这里只是演示 [PlanningScene](http://docs.ros.org/noetic/api/moveit_core/html/cpp/classplanning__scene_1_1PlanningScene.html) 类

- 例子中用到的碰撞检测：

  - [CollisionRequest](http://docs.ros.org/noetic/api/moveit_core/html/cpp/structcollision__detection_1_1CollisionRequest.html)：碰撞检测请求
  - [CollisionResult](http://docs.ros.org/noetic/api/moveit_core/html/cpp/structcollision__detection_1_1CollisionResult.html) ：碰撞检测结果
  - [AllowedCollisionMatrix](http://docs.ros.org/noetic/api/moveit_core/html/cpp/classcollision__detection_1_1AllowedCollisionMatrix.html) (ACM) ：提供了一种mechanism机制让我们忽略某些link的碰撞检测，即即使真的碰撞了也不报错。

- 例子中用到的约束检测：

  有两种：

  - 来自 [KinematicConstrain](http://docs.ros.org/noetic/api/moveit_core/html/cpp/classkinematic__constraints_1_1KinematicConstraint.html)类的：
    - [JointConstraint](http://docs.ros.org/noetic/api/moveit_core/html/cpp/classkinematic__constraints_1_1JointConstraint.html)
    - [PositionConstraint](http://docs.ros.org/noetic/api/moveit_core/html/cpp/classkinematic__constraints_1_1PositionConstraint.html)
    - [OrientationConstraint](http://docs.ros.org/noetic/api/moveit_core/html/cpp/classkinematic__constraints_1_1OrientationConstraint.html) 
    - [VisibilityConstraint](http://docs.ros.org/noetic/api/moveit_core/html/cpp/classkinematic__constraints_1_1VisibilityConstraint.html) 
    - 用于设置这些constraint的函数定义在 [utils.h](http://docs.ros.org/noetic/api/moveit_core/html/cpp/utils_8h.html) file

  - 用户自定义的Constraints。（通过callback()的方式）。即例子中15行的`bool stateFeasibilityTestExample`

- [代码](https://github.com/ros-planning/moveit_tutorials/tree/master/doc/planning_scene)

  ```c++
  #include <ros/ros.h>
  // MoveIt
  #include <moveit/robot_model_loader/robot_model_loader.h>
  #include <moveit/planning_scene/planning_scene.h>
  #include <moveit/kinematic_constraints/utils.h>
  
  
  //********************** stateFeasibilityTestExample **********************
  // User defined constraints can also be specified to the PlanningScene
  // class. This is done by specifying a callback using the
  // setStateFeasibilityPredicate function. Here's a simple example of a
  // user-defined callback that checks whether the "panda_joint1" of
  // the Panda robot is at a positive or negative angle:
  // 自定义一个关节joint的constraints/Feasibility可行性。如果可行就返回1，不可行就返回0.
  bool stateFeasibilityTestExample(const moveit::core::RobotState& kinematic_state, bool /*verbose*/)
  {
    const double* joint_values = kinematic_state.getJointPositions("panda_joint1");
    return (joint_values[0] > 0.0);
  }
  
  
  int main(int argc, char** argv)
  {
    ros::init(argc, argv, "panda_arm_kinematics");
    ros::AsyncSpinner spinner(1);
    spinner.start();
  
    //********************** Setup **********************
    // 加载模型
    robot_model_loader::RobotModelLoader robot_model_loader("robot_description");
    // 得到模型
    const moveit::core::RobotModelPtr& kinematic_model = robot_model_loader.getModel();
    // 创建一个PlanningScene类，注意这不推荐，更推荐使用PlanningSceneMonitor类
    planning_scene::PlanningScene planning_scene(kinematic_model);
  
    
    //********************** Collision Checking **********************
    // 一、Self-collision checking
    // ~~~~~~~~~~~~~~~~~~~~~~~
    // 检查机器人是否会自己撞自己。
    collision_detection::CollisionRequest collision_request;
    collision_detection::CollisionResult collision_result;
    // Self collision checking使用的是URDF中机器模型提供的collision mesh来检测。
    planning_scene.checkSelfCollision(collision_request, collision_result);
    ROS_INFO_STREAM("Test 1: Current state is " << (collision_result.collision ? "in" : "not in") << " self collision");
  
    // 二、Change the state
    // ~~~~~~~~~~~~~~~~
    // PlanningScene类的实例，保存着当前机器人的状态，读取并改变它。
    moveit::core::RobotState& current_state = planning_scene.getCurrentStateNonConst();
    // 设置机器人的状态为任意位置
    current_state.setToRandomPositions();
    // 在进行新的碰撞检测之前需要先清楚旧的碰撞请求
    collision_result.clear();
    planning_scene.checkSelfCollision(collision_request, collision_result);
    ROS_INFO_STREAM("Test 2: Current state is " << (collision_result.collision ? "in" : "not in") << " self collision");
  
    // 三、Checking for a group
    // ~~~~~~~~~~~~~~~~~~~~
    // 检查机器人 手 这一个部分会不会和机器人其他部分相撞
    collision_request.group_name = "hand";
    // 设置机器人的状态为任意位置
    current_state.setToRandomPositions();
    // 在进行新的碰撞检测之前需要先清楚旧的碰撞请求
    collision_result.clear();
    planning_scene.checkSelfCollision(collision_request, collision_result);
    ROS_INFO_STREAM("Test 3: Current state is " << (collision_result.collision ? "in" : "not in") << " self collision");
  
    // 四、Getting Contact Information
    // ~~~~~~~~~~~~~~~~~~~~
    // 将机器手调到一个边界内的位姿
    std::vector<double> joint_values = { 0.0, 0.0, 0.0, -2.9, 0.0, 1.4, 0.0 };
    const moveit::core::JointModelGroup* joint_model_group = current_state.getJointModelGroup("panda_arm");
    current_state.setJointGroupPositions(joint_model_group, joint_values);
    ROS_INFO_STREAM("Test 4: Current state is "
                    << (current_state.satisfiesBounds(joint_model_group) ? "valid" : "not valid"));
    // 获得机器人在给定配置下可能发生碰撞的信息
    collision_request.contacts = true;
    collision_request.max_contacts = 1000;
    collision_result.clear();
    planning_scene.checkSelfCollision(collision_request, collision_result);
    ROS_INFO_STREAM("Test 5: Current state is " << (collision_result.collision ? "in" : "not in") << " self collision");
    collision_detection::CollisionResult::ContactMap::const_iterator it;
    for (it = collision_result.contacts.begin(); it != collision_result.contacts.end(); ++it)
    {
      ROS_INFO("Contact between: %s and %s", it->first.first.c_str(), it->first.second.c_str());
    }
  
    // 五、Modifying the Allowed Collision Matrix
    // ~~~~~~~~~~~~~~~~~~~~  
    // 初始化要忽略碰撞的link矩阵，ACM
    collision_detection::AllowedCollisionMatrix acm = planning_scene.getAllowedCollisionMatrix();
    // 得到当前状态
    moveit::core::RobotState copied_state = planning_scene.getCurrentState();
    // 把当前碰撞的link都加入到矩阵ACM中去，用于忽略碰撞检测
    collision_detection::CollisionResult::ContactMap::const_iterator it2;
    for (it2 = collision_result.contacts.begin(); it2 != collision_result.contacts.end(); ++it2)
    {
      acm.setEntry(it2->first.first, it2->first.second, true);
    }
    collision_result.clear();
    // 最后一个参数是我们要忽略碰撞检测的link矩阵
    planning_scene.checkSelfCollision(collision_request, collision_result, copied_state, acm);
    ROS_INFO_STREAM("Test 6: Current state is " << (collision_result.collision ? "in" : "not in") << " self collision");
  
    // 六、Full Collision Checking
    // ~~~~~~~~~~~~~~~~~~~~~~~
    // checkCollision()不止检测机器人自己撞不撞，也会检测它和环境有无碰撞
    // checkCollision()不像checkSelfCollision(),他会填充pad机器人碰撞mesh
    collision_result.clear();
    planning_scene.checkCollision(collision_request, collision_result, copied_state, acm);
    ROS_INFO_STREAM("Test 7: Current state is " << (collision_result.collision ? "in" : "not in") << " self collision");
  
  
    //********************** Constraint Checking **********************
    // 一、Checking Kinematic Constraints
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // 获得所有link的名字
    std::string end_effector_name = joint_model_group->getLinkModelNames().back();
    geometry_msgs::PoseStamped desired_pose;
    desired_pose.pose.orientation.w = 1.0;
    desired_pose.pose.position.x = 0.3;
    desired_pose.pose.position.y = -0.185;
    desired_pose.pose.position.z = 0.5;
    desired_pose.header.frame_id = "panda_link0";
    // constructGoalConstraints()相关的文档在utils.h中，上面有链接
    // 为link组添加约束
    moveit_msgs::Constraints goal_constraint =
        kinematic_constraints::constructGoalConstraints(end_effector_name, desired_pose);
    // 更新link的位姿
    copied_state.setToRandomPositions();
    copied_state.update();
    // 用bool类型的方法来检测是否收到约束
    bool constrained = planning_scene.isStateConstrained(copied_state, goal_constraint);
    ROS_INFO_STREAM("Test 8: Random state is " << (constrained ? "constrained" : "not constrained"));
    // 1.1：相比于上面的一个更高效的方法：
    // 当我们像检查相同的约束over and over again一遍又一遍，我们可以首先创建KinematicConstraintSet类的实例
    // 它会预处理ROS约束信息，并将其设置为quick processing快速处理
    kinematic_constraints::KinematicConstraintSet kinematic_constraint_set(kinematic_model);
    kinematic_constraint_set.add(goal_constraint, planning_scene.getTransforms());
    bool constrained_2 = planning_scene.isStateConstrained(copied_state, kinematic_constraint_set);
    ROS_INFO_STREAM("Test 9: Random state is " << (constrained_2 ? "constrained" : "not constrained"));
    // 1.2： 1.1的另一种写法
    kinematic_constraints::ConstraintEvaluationResult constraint_eval_result =
        kinematic_constraint_set.decide(copied_state);
    ROS_INFO_STREAM("Test 10: Random state is " << (constraint_eval_result.satisfied ? "constrained" : "not constrained"));
  
    // 二、User-defined constraints
    // ~~~~~~~~~~~~~~~~~~~~~~~~
    // 利用PlanningScene的setStateFeasibilityPredicate()方法，将某一个boll类型的callback函数作为约束
    // stateFeasibilityTestExample是我们在最上面定义的boll类型的callback函数
    planning_scene.setStateFeasibilityPredicate(stateFeasibilityTestExample);
    // 用isStateFeasible()可以检测用户自定义约束
    bool state_feasible = planning_scene.isStateFeasible(copied_state);
    ROS_INFO_STREAM("Test 11: Random state is " << (state_feasible ? "feasible" : "not feasible"));
    // 最强的是isStateValid()，碰撞/约束/用户自定义约束，三个都会检测
    bool state_valid = planning_scene.isStateValid(copied_state, kinematic_constraint_set, "panda_arm");
    ROS_INFO_STREAM("Test 12: Random state is " << (state_valid ? "valid" : "not valid"));
  
  
    ros::shutdown();
    return 0;
  }
  
  ```

## 5.2 Planning Scene Monitor

- The [PlanningSceneMonitor](http://docs.ros.org/noetic/api/moveit_ros_planning/html/classplanning__scene__monitor_1_1PlanningSceneMonitor.html) is the recommended interface for maintaining an up-to-date planning scene.

- 不同类的关系

  - **RobotState**

    - The [RobotState](http://docs.ros.org/noetic/api/moveit_core/html/cpp/classmoveit_1_1core_1_1RobotState.html) is a snapshot简介 of a robot. It contains the [RobotModel](http://docs.ros.org/noetic/api/moveit_core/html/cpp/classmoveit_1_1core_1_1RobotModel.html) and a set of joint values.

  - **CurrentStateMonitor**

    - The [CurrentStateMonitor](http://docs.ros.org/noetic/api/moveit_ros_planning/html/classplanning__scene__monitor_1_1CurrentStateMonitor.html) (CSM) can be thought of as a ROS wrapper包装 for the RobotState.
    - It subscribes订阅 to a provided topic for [JointState](http://docs.ros.org/noetic/api/sensor_msgs/html/msg/JointState.html) messages that provide up-to-date最新的 sensor values for single degree of freedom actuators单自由度执行器, such as revolute旋转关节 or prismatic joints棱柱关节, and updates its internal RobotState with those joint values
    - o maintain up-to-date transform information for links and other frames attached with multiple-degree-of-freedom joints多自由度关节(floating and planar joints.), the CSM stores a TF2 [Buffer](http://docs.ros.org/noetic/api/tf2_ros/html/c++/classtf2__ros_1_1Buffer.html)缓冲器 that uses a TF2 [TransformListener](http://docs.ros.org/noetic/api/tf2_ros/html/c++/classtf2__ros_1_1TransformListener.html) to set their transforms in its internal data.

  - **Planning Scene**

    - The [PlanningScene](http://docs.ros.org/noetic/api/moveit_core/html/cpp/classplanning__scene_1_1PlanningScene.html) is a snapshot of the world that includes both the RobotState and any number of collision objects.
    - The Planning Scene can be used for collision checking as well as getting information about the environment.

  - **PlanningSceneMonitor**

    - The [PlanningSceneMonitor](http://docs.ros.org/noetic/api/moveit_ros_planning/html/classplanning__scene__monitor_1_1PlanningSceneMonitor.html) wraps a PlanningScene with ROS interfaces for keeping the PlanningScene up to date.用ROS包装了下达到实时更新Planning Scene的目的。To access the PlanningSceneMonitor’s underlying PlanningScene, use the provided [LockedPlanningSceneRW](http://docs.ros.org/noetic/api/moveit_ros_planning/html/classplanning__scene__monitor_1_1LockedPlanningSceneRW.html) and [LockedPlanningSceneRO](http://docs.ros.org/noetic/api/moveit_ros_planning/html/classplanning__scene__monitor_1_1LockedPlanningSceneRO.html) classes.

    - The PlanningSceneMonitor has the following objects对象, which have their own ROS interfaces for keeping sub-components of the planning scene up to date:

      - A [CurrentStateMonitor](http://docs.ros.org/noetic/api/moveit_ros_planning/html/classplanning__scene__monitor_1_1CurrentStateMonitor.html) for tracking updates to the RobotState via a `robot_state_subscriber_` and a `tf_buffer_`, as well as a planning scene subscriber for listening to planning scene diffs from other publishers.

      - An OccupancyMapMonitor for tracking updates to an OccupancyMap via ROS topics and services.

    - The PlanningSceneMonitor has the following subscribers订阅者:

      - `collision_object_subscriber_` - Listens to a provided topic for [CollisionObject](http://docs.ros.org/noetic/api/moveit_msgs/html/msg/CollisionObject.html) messages that might add, remove, or modify collision objects in the planning scene and passes them into its monitored planning scene

      - `planning_scene_world_subscriber_` - Listens to a provided topic for [PlanningSceneWorld](http://docs.ros.org/noetic/api/moveit_msgs/html/msg/PlanningSceneWorld.html) messages that may contain collision object information and/or octomap information. This is useful for keeping planning scene monitors in sync同步

      - `attached_collision_object_subscriber_` - Listens on a provided topic for [AttachedCollisionObject](http://docs.ros.org/noetic/api/moveit_msgs/html/msg/AttachedCollisionObject.html) messages that determine the attaching固定/detaching分离 of objects to links in the robot state.

    - The PlanningSceneMonitor has the following services服务:

      - `get_scene_service_` - Which is an optional service to get the full planning scene state.

    - The PlanningSceneMonitor is initialized with:

      - `startSceneMonitor` - Which starts the `planning_scene_subscriber_`,

      - `startWorldGeometryMonitor` - Which starts the `collision_object_subscriber_`, the `planning_scene_world_subscriber_`, and the OccupancyMapMonitor,

      - `startStateMonitor` - Which starts the CurrentStateMonitor and the `attached_collision_object_subscriber_`,

      - `startPublishingPlanningScene` - Which starts another thread for publishing the entire planning scene on a provided topic for other PlanningSceneMonitors to subscribe to, and

      - `providePlanningSceneService` - Which starts the `get_scene_service_`.

- PlanningSceneInterface

  - The [PlanningSceneInterface](http://docs.ros.org/noetic/api/moveit_ros_planning_interface/html/classmoveit_1_1planning__interface_1_1PlanningSceneInterface.html) is a useful class for publishing updates to a MoveGroup’s [PlanningSceneMonitor](http://docs.ros.org/noetic/api/moveit_ros_planning/html/classplanning__scene__monitor_1_1PlanningSceneMonitor.html) through a C++ API without creating your own subscribers and service clients. It may not work without MoveGroup or MoveItCpp.

## 5.3 Planning Scene ROS API

- 使用Planning Scene可以在world中添加或删除物体，固定attaching或分离detaching物体objects到机器人上

- 运行：

  - 第一个shell:`roslaunch panda_moveit_config demo.launch`
  - 第二个shell:`roslaunch moveit_tutorials planning_scene_ros_api_tutorial.launch`

- 一些代码库

  - [moveit_visual_tools](http://docs.ros.org/en/kinetic/api/moveit_visual_tools/html/classmoveit__visual__tools_1_1MoveItVisualTools.html): built in top of(继承于) [rviz_visual_tools](http://docs.ros.org/en/kinetic/api/rviz_visual_tools/html/classrviz__visual__tools_1_1RvizVisualTools.html)。用于在Rviz中输出标记markers,轨迹trajectories和碰撞物collision objects
  
- [代码](https://github.com/ros-planning/moveit_tutorials/tree/master/doc/planning_scene_ros_api)：

  ```c++
  #include <ros/ros.h>
  #include <geometry_msgs/Pose.h>
  // MoveIt
  #include <moveit_msgs/PlanningScene.h>
  #include <moveit_msgs/AttachedCollisionObject.h>
  #include <moveit_msgs/GetStateValidity.h>
  #include <moveit_msgs/DisplayRobotState.h>
  #include <moveit_msgs/ApplyPlanningScene.h>
  #include <moveit/robot_model_loader/robot_model_loader.h>
  #include <moveit/robot_state/robot_state.h>
  #include <moveit/robot_state/conversions.h>
  #include <moveit_visual_tools/moveit_visual_tools.h>
  
  int main(int argc, char** argv)
  {
    ros::init(argc, argv, "planning_scene_ros_api_tutorial");
    ros::AsyncSpinner spinner(1);
    spinner.start();
    ros::NodeHandle node_handle;
      
    //********************** Visualization **********************
    // The package MoveItVisualTools provides many capabilities能力 for visualizing objects, robots, and trajectories in RViz as well as debugging tools such as step-by-step introspection内省 of a script脚本.
    moveit_visual_tools::MoveItVisualTools visual_tools("panda_link0");
    // 让Rviz删除所有的标记markers
    visual_tools.deleteAllMarkers();
  
  
    //********************** ROS API **********************
    // The ROS API to the planning scene publisher is through a topic interface using "diffs". 
    // A planning scene diff is the difference between the current planning scene (maintained by the move_group node) and the new planning scene desired by the user.
    // ROS API对规划场景发布消息是使用“diffs”通过一个话题topic接口实现的。
    // planning scene diff是当前规划场景与用户使用的新的规划场景不同的地方。
    ros::Publisher planning_scene_diff_publisher = node_handle.advertise<moveit_msgs::PlanningScene>("planning_scene", 1);
    // 墙上时钟
    ros::WallDuration sleep_t(0.5);
    while (planning_scene_diff_publisher.getNumSubscribers() < 1)
    {
      sleep_t.sleep();
    }
    // 等待用户反馈（通过一个按钮button/操纵杆joystick）
    visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to start the demo");
  
    
    //********************** Define the attached object message **********************
    // 使用这个message来从世界中添加或去除subtract物体，并将物体连接到机器人上
    moveit_msgs::AttachedCollisionObject attached_object;
    // collisionObject将会通过一个fixed joint来附加attach到这个link
    attached_object.link_name = "panda_hand";
    // 与此object相关联的TF Frame
    attached_object.object.header.frame_id = "panda_hand";
    // The id of the object 
    attached_object.object.id = "box";
    // object的初始位置
    geometry_msgs::Pose pose;
    pose.position.z = 0.11;
    pose.orientation.w = 1.0;
    // 定义object的geometric primitives几何图元
    shape_msgs::SolidPrimitive primitive;
    primitive.type = primitive.BOX;
    primitive.dimensions.resize(3);
    primitive.dimensions[0] = 0.075;
    primitive.dimensions[1] = 0.075;
    primitive.dimensions[2] = 0.075;
    // 将geometric primitives几何图元加到attached_objects这一messages中去
    attached_object.object.primitives.push_back(primitive);
    attached_object.object.primitive_poses.push_back(pose);
    // 定义这个物体添加到机器人的方式，这里是ADD
    attached_object.object.operation = attached_object.object.ADD;
    // 物体可以触摸touch的links set.即让object和robot指定links之间的collision检测消失
    attached_object.touch_links = std::vector<std::string>{ "panda_hand", "panda_leftfinger", "panda_rightfinger" };
  
    
    //********************** Add an object into the environment **********************
    ROS_INFO("Adding the object into the world at the location of the hand.");
    moveit_msgs::PlanningScene planning_scene;
    // 将object加入到世界，即planning scene.
    planning_scene.world.collision_objects.push_back(attached_object.object);
    // 一个Flag,表明indicate是否这个scene被认为是关于一些其他scene的diff（）
    // planning scene diff是当前规划场景与用户使用的新的规划场景不同的地方。
    planning_scene.is_diff = true;
    // 将场景变化发送给rviz
    planning_scene_diff_publisher.publish(planning_scene);
    visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to continue the demo");
  
  
    //********************** Synchronous vs Asynchronous updates **********************
    // 有两个独立的机制separate mechanisms用于和使用diffs的move_group来interact沟通。
    // 1.synchronous update同步更新：Send a diff via a rosservice call and block until the diff is applied 
    // 2.asynchronous update异步更新：Send a diff via a topic,continue even though the diff might not be applied yet
    // 前面是用topic来传输，下面用service来传输
    // 创建一个服务
    ros::ServiceClient planning_scene_diff_client = node_handle.serviceClient<moveit_msgs::ApplyPlanningScene>("apply_planning_scene");
    // 堵塞block直到service被广播advertise和可达available
    planning_scene_diff_client.waitForExistence();
    // and send the diffs to the planning scene via a service call:
    // 将信息发送并调用服务
    moveit_msgs::ApplyPlanningScene srv;
    srv.request.scene = planning_scene;
    planning_scene_diff_client.call(srv);
    // Note that this does not continue until we are sure the diff has been applied.
  
    //********************** Attach an object to the robot **********************
    // 当机器人从环境中拾取一个物体时，我们需要将物体attach附加到机器人上。以便处理机器人模型的任何组件都能考虑到附加的物体（比如碰撞检测）
    // 附加attach一个object物体到机器人需要2步操作
    // 1.从环境environment中消除原物体
    moveit_msgs::CollisionObject remove_object;
    remove_object.id = "box";
    remove_object.header.frame_id = "panda_hand";
    remove_object.operation = remove_object.REMOVE;
    ROS_INFO("Attaching the object to the hand and removing it from the world.");
    planning_scene.world.collision_objects.clear();
    planning_scene.world.collision_objects.push_back(remove_object);
    // 2.将物体添加到机器人robot
    planning_scene.robot_state.attached_collision_objects.push_back(attached_object);
    planning_scene.robot_state.is_diff = true;
    planning_scene_diff_publisher.publish(planning_scene);
    visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to continue the demo");
  
    
    //********************** Detach an object from the robot **********************
    // Detaching an object from the robot requires two operations
    // 从机器人身上分离一个物体要2个步骤
    // 1.从机器人身上分离物体
    moveit_msgs::AttachedCollisionObject detach_object;
    detach_object.object.id = "box";
    detach_object.link_name = "panda_hand";
    detach_object.object.operation = attached_object.object.REMOVE;
    ROS_INFO("Detaching the object from the robot and returning it to the world.");
    planning_scene.robot_state.attached_collision_objects.clear();
    planning_scene.robot_state.attached_collision_objects.push_back(detach_object);
    planning_scene.robot_state.is_diff = true;
    // 2.将物体重新放入世界环境中
    planning_scene.world.collision_objects.clear();
    planning_scene.world.collision_objects.push_back(attached_object.object);
    planning_scene.is_diff = true;
    planning_scene_diff_publisher.publish(planning_scene);
    visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to continue the demo");
  
  
    //********************** Remove the object from the collision world **********************
    // 将物体从世界环境中移除
    ROS_INFO("Removing the object from the world.");
    planning_scene.robot_state.attached_collision_objects.clear();
    planning_scene.world.collision_objects.clear();
    planning_scene.world.collision_objects.push_back(remove_object);
    planning_scene_diff_publisher.publish(planning_scene);
    // END_TUTORIAL
  
    visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to end the demo");
    ros::shutdown();
    return 0;
  }
  ```


# 六、Motion Planning

## 6.1 Motion Planning API

- Planner规划器是作为plugin插件集成在MoveIt里的，所以可以用**ROS Pluginlib interface**来加载任何planner。

  在加载前需要2个Object 

  - [RobotModel](http://docs.ros.org/en/noetic/api/moveit_core/html/cpp/classmoveit_1_1core_1_1RobotModel.html): 通过实例化instantiating一个[RobotModelLoader](http://docs.ros.org/noetic/api/moveit_ros_planning/html/classrobot__model__loader_1_1RobotModelLoader.html) object来查看 robot description描述 on the ROS parameter server 并构建 RobotModel
  - [PlanningScene](http://docs.ros.org/noetic/api/moveit_core/html/cpp/classplanning__scene_1_1PlanningScene.html): 通过上面的RobotModel来构建PlanningScene

  ROS Pluginlib interface的相关资料

  - [ROS官网资料](http://wiki.ros.org/pluginlib)
  - MoveIt的[Planning_Interface](http://docs.ros.org/en/jade/api/moveit_ros_planning_interface/html/classmoveit_1_1planning__interface_1_1MoveGroup.html)

- 两个shell运行
  - `roslaunch panda_moveit_config demo.launch`
  - `roslaunch moveit_tutorials motion_planning_api_tutorial.launch`

- [代码](https://github.com/ros-planning/moveit_tutorials/tree/master/doc/motion_planning_api)

  ```c++
  #include <pluginlib/class_loader.h>
  #include <ros/ros.h>
  #include <moveit/robot_model_loader/robot_model_loader.h>
  #include <moveit/planning_interface/planning_interface.h>
  #include <moveit/planning_scene/planning_scene.h>
  #include <moveit/kinematic_constraints/utils.h>
  #include <moveit_msgs/DisplayTrajectory.h>
  #include <moveit_msgs/PlanningScene.h>
  #include <moveit_visual_tools/moveit_visual_tools.h>
  #include <boost/scoped_ptr.hpp>
  
  int main(int argc, char** argv)
  {
    const std::string node_name = "motion_planning_tutorial";
    ros::init(argc, argv, node_name);
    ros::AsyncSpinner spinner(1);
    spinner.start();
    ros::NodeHandle node_handle("~");
  
  
    //********************** Start **********************
    const std::string PLANNING_GROUP = "panda_arm";
    // 创建RobotModelLoader的实例来从机器人描述中读取模型
    robot_model_loader::RobotModelLoader robot_model_loader("robot_description");
    // 通过RobotModelLoader的实例来创建RobotModel的实例，加载机器人模型
    const moveit::core::RobotModelPtr& robot_model = robot_model_loader.getModel();
    // 创建一个RobotState实例来追踪机器人的状态位姿
    moveit::core::RobotStatePtr robot_state(new moveit::core::RobotState(robot_model));
    // 得到一个特定关节组joint group的模型（这里是panda_arm组），即我们要控制哪些joint
    const moveit::core::JointModelGroup* joint_model_group = robot_state->getJointModelGroup(PLANNING_GROUP);
    // 通过RobotModel的实例来创建Planning_scene的实例，它保存了整个世界的状态包括机器人的
    planning_scene::PlanningScenePtr planning_scene(new planning_scene::PlanningScene(robot_model));
    // Configure a valid robot state
    planning_scene->getCurrentStateNonConst().setToDefaultValues(joint_model_group, "ready");
      
    // 使用ROS pluginlib library来加载planner规划器
    boost::scoped_ptr<pluginlib::ClassLoader<planning_interface::PlannerManager>> planner_plugin_loader;
    planning_interface::PlannerManagerPtr planner_instance;
    std::string planner_plugin_name;
  
    // 从ROS parameter server得到我们希望加载的planner的名字，并加载planner和抓取所有的例外exception
    if (!node_handle.getParam("planning_plugin", planner_plugin_name))
      ROS_FATAL_STREAM("Could not find planner plugin name");
    try
    {
      planner_plugin_loader.reset(new pluginlib::ClassLoader<planning_interface::PlannerManager>(
          "moveit_core", "planning_interface::PlannerManager"));
    }
    catch (pluginlib::PluginlibException& ex)
    {
      ROS_FATAL_STREAM("Exception while creating planning plugin loader " << ex.what());
    }
    try
    {
      planner_instance.reset(planner_plugin_loader->createUnmanagedInstance(planner_plugin_name));
      if (!planner_instance->initialize(robot_model, node_handle.getNamespace()))
        ROS_FATAL_STREAM("Could not initialize planner instance");
      ROS_INFO_STREAM("Using planning interface '" << planner_instance->getDescription() << "'");
    }
    catch (pluginlib::PluginlibException& ex)
    {
      const std::vector<std::string>& classes = planner_plugin_loader->getDeclaredClasses();
      std::stringstream ss;
      for (const auto& cls : classes)
        ss << cls << " ";
      ROS_ERROR_STREAM("Exception while loading planner '" << planner_plugin_name << "': " << ex.what() << std::endl
                                                           << "Available plugins: " << ss.str());
    }
  
    
    //********************** Visualization **********************
    // 使用MoveItVisualTools在Rviz中可视化物体，机器人和轨迹。5.3中同样用到了，该工具同样可用于debug
    namespace rvt = rviz_visual_tools;
    moveit_visual_tools::MoveItVisualTools visual_tools("panda_link0");
    visual_tools.loadRobotStatePub("/display_robot_state");
    visual_tools.enableBatchPublishing();
    visual_tools.deleteAllMarkers();  // clear all old markers
    visual_tools.trigger();
    // 通过在Rviz中用按钮、键盘等来远程控制
    visual_tools.loadRemoteControl();
    // Rviz中有多样的markers标记，这里使用text,cylinder圆柱和spheres球体
    Eigen::Isometry3d text_pose = Eigen::Isometry3d::Identity();
    text_pose.translation().z() = 1.75;
    visual_tools.publishText(text_pose, "Motion Planning API Demo", rvt::WHITE, rvt::XLARGE);
    /* Batch publishing is used to reduce the number of messages being sent to RViz for large visualizations */
    // Trigger触发 the publish function to send out all collected markers.
    visual_tools.trigger();
    /* We can also use visual_tools to wait for user input */
    visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to start the demo");
  
  
    //********************** Pose Goal **********************
    // 下面要给panda arm的end-effector末端执行器设置一个desired pose。并为其创建一个motion plan
    // 首先输出当前状态
    visual_tools.publishRobotState(planning_scene->getCurrentStateNonConst(), rviz_visual_tools::GREEN);
    // Trigger触发 the publish function to send out all collected markers.
    visual_tools.trigger();
    // 这是一个msg,查看文档planning_interface namespace
    planning_interface::MotionPlanRequest req;
    // 这是一个struct
    planning_interface::MotionPlanResponse res;
    // 目标位姿设置
    geometry_msgs::PoseStamped pose;
    pose.header.frame_id = "panda_link0";
    pose.pose.position.x = 0.3;
    pose.pose.position.y = 0.4;
    pose.pose.position.z = 0.75;
    pose.pose.orientation.w = 1.0;
    // A tolerance偏差 of 0.01 m is specified in position and 0.01 radians in orientation
    // 3表示Z轴
    std::vector<double> tolerance_pose(3, 0.01);
    std::vector<double> tolerance_angle(3, 0.01);
    // 使用kinematic_constraints包来将request构建为一个constraint
    // 这里用的constructGoalConstraints()第五个重构，1.施加限制的linke名，2.link位姿，3.施加的PositionConstraint位置限制,4.施加的OrientationConstraint方向限制
    moveit_msgs::Constraints pose_goal = kinematic_constraints::constructGoalConstraints("panda_link8", pose, tolerance_pose, tolerance_angle);
    req.group_name = PLANNING_GROUP;
    // 将上面设置的限制pose_goal添加到req这一msg中去。
    // req.goal_constraints有1.joint_constraints 2.position_constraints 3.orientation_constraints 4.visibility_constraints
    // 上面pose_goal只定义了2.position_constraints 3.orientation_constraints
    req.goal_constraints.push_back(pose_goal);
  
    // We now construct a planning context that encapsulate概述 the scene, the request and the response. We call the planner using this planning context
    // 使用planner_instance类的getPlanningContex()方法。
    // 它会在给定的场景planning_scene和规划请求req下来构建上下文context。如果失败，运行error code并返回空指针empty ptr。
    // 每次运动规划（构建上下文）都会从头开始
    planning_interface::PlanningContextPtr context = planner_instance->getPlanningContext(planning_scene, req, res.error_code_);
    // 解决运动规划问题并将结果保存在res中
    context->solve(res);
    if (res.error_code_.val != res.error_code_.SUCCESS)
    {
      ROS_ERROR("Could not compute plan successfully");
      return 0;
    }
  
      
    //********************** Visualize the result **********************
    // 发送moveit_msgs::DisplayTrajectory类型的消息msg到话题"/move_group/display_planned_path"
    ros::Publisher display_publisher = node_handle.advertise<moveit_msgs::DisplayTrajectory>("/move_group/display_planned_path", 1, true);
    moveit_msgs::DisplayTrajectory display_trajectory;
    // 设置要展示的路径规划
    moveit_msgs::MotionPlanResponse response;
    res.getMessage(response);
    display_trajectory.trajectory_start = response.trajectory_start;
    display_trajectory.trajectory.push_back(response.trajectory);
    // 展示末端执行器的路径线
    visual_tools.publishTrajectoryLine(display_trajectory.trajectory.back(), joint_model_group);
    visual_tools.trigger();
    // 把展示信息放给rviz
    display_publisher.publish(display_trajectory);
    // 将planning scene中的状态设置为规划路径的最后状态
    robot_state->setJointGroupPositions(joint_model_group, response.trajectory.joint_trajectory.points.back().positions);
    planning_scene->setCurrentState(*robot_state.get());
    // 展示目标状态
    visual_tools.publishRobotState(planning_scene->getCurrentStateNonConst(), rviz_visual_tools::GREEN);
    visual_tools.publishAxisLabeled(pose.pose, "goal_1");
    visual_tools.publishText(text_pose, "Pose Goal (1)", rvt::WHITE, rvt::XLARGE);
    visual_tools.trigger();
    /* We can also use visual_tools to wait for user input */
    visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to continue the demo");
  
  
    //********************** Joint Space Goals **********************
    // Now, setup a joint space goal
    // 创建一个moveit::core::RobotState的实例，它是模型robot_model的状态
    moveit::core::RobotState goal_state(robot_model);
    std::vector<double> joint_values = { -1.0, 0.7, 0.7, -1.5, -0.7, 2.0, 0.0 };
    // 将我们的目标状态joint_values，设置到我们要控制的关节组joint_model_group
    goal_state.setJointGroupPositions(joint_model_group, joint_values);
    // 设置关节限制
    moveit_msgs::Constraints joint_goal = kinematic_constraints::constructGoalConstraints(goal_state, joint_model_group);
    req.goal_constraints.clear();
    req.goal_constraints.push_back(joint_goal);
  
    // Call the planner and visualize the trajectory
    // 重新构建规划上下文
    context = planner_instance->getPlanningContext(planning_scene, req, res.error_code_);
    // 解决运动规划问题并将结果保存在res中
    context->solve(res);
    /* Check that the planning was successful */
    if (res.error_code_.val != res.error_code_.SUCCESS)
    {
      ROS_ERROR("Could not compute plan successfully");
      return 0;
    }
    // 可视化轨迹
    res.getMessage(response);
    display_trajectory.trajectory.push_back(response.trajectory);
    // 现在可以看到连续的2段规划路径
    visual_tools.publishTrajectoryLine(display_trajectory.trajectory.back(), joint_model_group);
    visual_tools.trigger();
    display_publisher.publish(display_trajectory);
    // 将planning scene中的状态设置为规划路径的最后状态
    robot_state->setJointGroupPositions(joint_model_group, response.trajectory.joint_trajectory.points.back().positions);
    planning_scene->setCurrentState(*robot_state.get());
    // Display the goal state
    visual_tools.publishRobotState(planning_scene->getCurrentStateNonConst(), rviz_visual_tools::GREEN);
    visual_tools.publishAxisLabeled(pose.pose, "goal_2");
    visual_tools.publishText(text_pose, "Joint Space Goal (2)", rvt::WHITE, rvt::XLARGE);
    visual_tools.trigger();
    /* Wait for user input */
    visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to continue the demo");
  
  
    // 再用上面的pose_goal 让末端执行器回到原处
    req.goal_constraints.clear();
    req.goal_constraints.push_back(pose_goal);
    context = planner_instance->getPlanningContext(planning_scene, req, res.error_code_);
    context->solve(res);
    res.getMessage(response);
    // 显示轨迹
    display_trajectory.trajectory.push_back(response.trajectory);
    visual_tools.publishTrajectoryLine(display_trajectory.trajectory.back(), joint_model_group);
    visual_tools.trigger();
    display_publisher.publish(display_trajectory);
    // 将planning scene中的状态设置为规划路径的最后状态
    robot_state->setJointGroupPositions(joint_model_group, response.trajectory.joint_trajectory.points.back().positions);
    planning_scene->setCurrentState(*robot_state.get());
    // Display the goal state
    visual_tools.publishRobotState(planning_scene->getCurrentStateNonConst(), rviz_visual_tools::GREEN);
    visual_tools.trigger();
    /* Wait for user input */
    visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to continue the demo");
  
    //********************** Adding Path Constraints **********************
    // 添加一个新的pose goal
    pose.pose.position.x = 0.32;
    pose.pose.position.y = -0.25;
    pose.pose.position.z = 0.65;
    pose.pose.orientation.w = 1.0;
    moveit_msgs::Constraints pose_goal_2 = kinematic_constraints::constructGoalConstraints("panda_link8", pose, tolerance_pose, tolerance_angle);
    // 尝试移动到新的位姿
    req.goal_constraints.clear();
    req.goal_constraints.push_back(pose_goal_2);
    /* But, let's impose添加 a path constraint on the motion.
       Here, we are asking for the end-effector to stay level*/
    geometry_msgs::QuaternionStamped quaternion;
    quaternion.header.frame_id = "panda_link0";
    quaternion.quaternion.w = 1.0;
    req.path_constraints = kinematic_constraints::constructGoalConstraints("panda_link8", quaternion);
    // 定义一个允许的规划空间
    // Imposing path constraints requires the planner to reason推理 in the space of possible positions of the end-effector末端执行器的可能位置空间(the workspace of the robot)
    // because of this, we need to specify a bound指定一个界限 for the allowed planning volume允许的规划体积 as well也;
    // Note: a default bound默认边界 is automatically filled by the WorkspaceBounds request adapter (part of the OMPL pipeline,but that is not being used in this example).
    // We use a bound that definitely一定 includes the reachable space for the arm. This is fine because sampling is not done in this volume when planning for the arm; the bounds are only used to determine确定 if the sampled configurations采样配置 are valid.
    req.workspace_parameters.min_corner.x = req.workspace_parameters.min_corner.y =
        req.workspace_parameters.min_corner.z = -2.0;
    req.workspace_parameters.max_corner.x = req.workspace_parameters.max_corner.y =
        req.workspace_parameters.max_corner.z = 2.0;
  
    // Call the planner and visualize all the plans created so far.
    context = planner_instance->getPlanningContext(planning_scene, req, res.error_code_);
    context->solve(res);
    res.getMessage(response);
    display_trajectory.trajectory.push_back(response.trajectory);
    visual_tools.publishTrajectoryLine(display_trajectory.trajectory.back(), joint_model_group);
    visual_tools.trigger();
    display_publisher.publish(display_trajectory);
    // 将planning scene中的状态设置为规划路径的最后状态
    robot_state->setJointGroupPositions(joint_model_group, response.trajectory.joint_trajectory.points.back().positions);
    planning_scene->setCurrentState(*robot_state.get());
    // Display the goal state
    visual_tools.publishRobotState(planning_scene->getCurrentStateNonConst(), rviz_visual_tools::GREEN);
    visual_tools.publishAxisLabeled(pose.pose, "goal_3");
    visual_tools.publishText(text_pose, "Orientation Constrained Motion Plan (3)", rvt::WHITE, rvt::XLARGE);
    visual_tools.trigger();
  
    // END_TUTORIAL
    /* Wait for user input */
    visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to exit the demo");
    planner_instance.reset();
  
    return 0;
  }
  ```

  

## 6.2 Motion Planning Pipeline

- 在 MoveIt 中，运动规划器motion planner被设置为规划路径plan path。 但是，有时我们可能希望对运动规划请求motion planning request进行预处理pre-process或对规划路径planned path进行后处理post process（例如，用于时间参数化time parameterization）。 在这种情况下，我们使用规划管道planning pipeline来将运动规划器与预处理pre-process和后处理post process阶段链接起来chain。 预处理和后处理阶段，称为计划请求适配器olanning request adapters，可以通过 ROS 参数服务器parameter server进行配置configure。

- ***Planning Pipeline的主要作用就是，在 ROS 服务器中设置参数，在进行轨迹规划时，能够同时+自动进行前处理和后处理。\***

- 在2个shell中运行

  - roslaunch panda_moveit_config demo.launch
  - roslaunch moveit_tutorials motion_planning_pipeline_tutorial.launch

- 代码功能：

  - The robot moves its right arm to the pose goal in front of it,
  - The robot moves its right arm to the joint goal to the side,
  - The robot moves its right arm back to the original pose goal in front of it,

- [代码](https://github.com/ros-planning/moveit_tutorials/tree/master/doc/motion_planning_pipeline)

  - 和Planner规划器一样，加载planning pipeline需要2个Object 

    - [RobotModel](http://docs.ros.org/en/noetic/api/moveit_core/html/cpp/classmoveit_1_1core_1_1RobotModel.html): 通过实例化instantiating一个[RobotModelLoader](http://docs.ros.org/noetic/api/moveit_ros_planning/html/classrobot__model__loader_1_1RobotModelLoader.html) object来查看 robot description描述 on the ROS parameter server 并构建 RobotModel
    - [PlanningScene](http://docs.ros.org/noetic/api/moveit_core/html/cpp/classplanning__scene_1_1PlanningScene.html): 通过上面的RobotModel来构建PlanningScene

  ```c++
  #include <pluginlib/class_loader.h>
  #include <ros/ros.h>
  #include <moveit/robot_model_loader/robot_model_loader.h>
  #include <moveit/robot_state/conversions.h>
  #include <moveit/planning_pipeline/planning_pipeline.h>
  #include <moveit/planning_interface/planning_interface.h>
  #include <moveit/planning_scene_monitor/planning_scene_monitor.h>
  #include <moveit/kinematic_constraints/utils.h>
  #include <moveit_msgs/DisplayTrajectory.h>
  #include <moveit_msgs/PlanningScene.h>
  #include <moveit_visual_tools/moveit_visual_tools.h>
  
  int main(int argc, char** argv)
  {
    ros::init(argc, argv, "move_group_tutorial");
    ros::AsyncSpinner spinner(1);
    spinner.start();
    ros::NodeHandle node_handle("~");
  
    //********************** Start **********************
    // 创建RobotModelLoaderPtr实例，读取模型
    robot_model_loader::RobotModelLoaderPtr robot_model_loader(new robot_model_loader::RobotModelLoader("robot_description"));
    // 和6.1中不一样，这里用的是监视器PlanningSceneMonitorPtr来创建planningscene实例
    planning_scene_monitor::PlanningSceneMonitorPtr psm(
        new planning_scene_monitor::PlanningSceneMonitor(robot_model_loader));
  
    // Start the scene monitor (ROS topic-based)
    // listen for planning scene messages on topic /XXX and apply them to the internal planning scene accordingly相应的
    psm->startSceneMonitor();
    // Start the OccupancyMapMonitor占用地图
    // listens to changes of world geometry几何结构, collision objects, and (optionally) octomaps
    // octomap是一种基于八叉树的三维地图创建工具
    psm->startWorldGeometryMonitor();
    // Start the current state monitor.
    // listen to joint state updates as well as changes in attached collision objects and update the internal planning scene accordingly
    psm->startStateMonitor();
    // We can also use the RobotModelLoader to get a robot model which contains the robot's kinematic information
    moveit::core::RobotModelPtr robot_model = robot_model_loader->getModel();
  
    // We can get the most up to date robot state from the PlanningSceneMonitor by locking锁住 the internal planning scene for reading. This lock锁 ensures that the underlying scene isn't updated while we are reading it's state锁保证了读状态时不会更新当前scene.
    //  RobotState's are useful for computing the forward and inverse kinematics of the robot among many other uses 
    moveit::core::RobotStatePtr robot_state(
        new moveit::core::RobotState(planning_scene_monitor::LockedPlanningSceneRO(psm)->getCurrentState()));
  
    // 建立一个JointModelGroup实例来追踪当前机器人位姿和规划
    // The Joint Model group is useful for dealing with one set of joints at a time such as a left arm or a end effector
    const moveit::core::JointModelGroup* joint_model_group = robot_state->getJointModelGroup("panda_arm");
  
    // 创建PlanningPipeline的实例，它使用ROS parameter server来确定一系列的resquest adapters和planning plugin
    planning_pipeline::PlanningPipelinePtr planning_pipeline(
        new planning_pipeline::PlanningPipeline(robot_model, node_handle, "planning_plugin", "request_adapters"));
  
   
    //********************** Visualization **********************
    namespace rvt = rviz_visual_tools;
    moveit_visual_tools::MoveItVisualTools visual_tools("panda_link0");
    visual_tools.deleteAllMarkers();
    // remote control（按钮或键盘控制输入）
    visual_tools.loadRemoteControl();
    // RViz provides many types of markers, in this demo we will use text, cylinders, and spheres
    Eigen::Isometry3d text_pose = Eigen::Isometry3d::Identity();
    text_pose.translation().z() = 1.75;
    visual_tools.publishText(text_pose, "Motion Planning Pipeline Demo", rvt::WHITE, rvt::XLARGE);
    // Batch publishing is used to reduce the number of messages being sent to RViz for large visualizations
    visual_tools.trigger();
    // We can also use visual_tools to wait for user input
    visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to start the demo");
  
  
    //********************** Pose Goal **********************
    // 和6.1中一样设置pose goal
    planning_interface::MotionPlanRequest req;
    planning_interface::MotionPlanResponse res;
    // 四元数位姿
    geometry_msgs::PoseStamped pose;
    pose.header.frame_id = "panda_link0";
    pose.pose.position.x = 0.3;
    pose.pose.position.y = 0.0;
    pose.pose.position.z = 0.75;
    pose.pose.orientation.w = 1.0;
  
    // A tolerance of 0.01 m is specified in position and 0.01 radians in orientation
    // 设置容忍误差，3表示z轴
    std::vector<double> tolerance_pose(3, 0.01);
    std::vector<double> tolerance_angle(3, 0.01);
  
    // 利用kinematic_constraints设置约束条件
    req.group_name = "panda_arm";
    moveit_msgs::Constraints pose_goal =
        kinematic_constraints::constructGoalConstraints("panda_link8", pose, tolerance_pose, tolerance_angle);
    req.goal_constraints.push_back(pose_goal);
  
    // Before planning, we will need a Read Only lock on the planning scene so that it does not modify修改 the world representation描述 while planning
    {
      planning_scene_monitor::LockedPlanningSceneRO lscene(psm);
      // Now, call the pipeline and check whether planning was successful. 
      planning_pipeline->generatePlan(lscene, req, res);
    }
    // Check that the planning was successful 
    if (res.error_code_.val != res.error_code_.SUCCESS)
    {
      ROS_ERROR("Could not compute plan successfully");
      return 0;
    }
  
    
    //********************** Visualize the result **********************
    ros::Publisher display_publisher =
        node_handle.advertise<moveit_msgs::DisplayTrajectory>("/move_group/display_planned_path", 1, true);
    moveit_msgs::DisplayTrajectory display_trajectory;
  
    // Visualize the trajectory 
    ROS_INFO("Visualizing the trajectory");
    moveit_msgs::MotionPlanResponse response;
    res.getMessage(response);
    display_trajectory.trajectory_start = response.trajectory_start;
    display_trajectory.trajectory.push_back(response.trajectory);
    display_publisher.publish(display_trajectory);
    visual_tools.publishTrajectoryLine(display_trajectory.trajectory.back(), joint_model_group);
    visual_tools.trigger();
    // Wait for user input
    visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to continue the demo");
  
  
    //********************** Joint Space Goals **********************
    /* First, set the state in the planning scene to the final state of the last plan */
    robot_state = planning_scene_monitor::LockedPlanningSceneRO(psm)->getCurrentStateUpdated(response.trajectory_start);
    robot_state->setJointGroupPositions(joint_model_group, response.trajectory.joint_trajectory.points.back().positions);
    moveit::core::robotStateToRobotStateMsg(*robot_state, req.start_state);
    // Now, setup a joint space goal
    moveit::core::RobotState goal_state(*robot_state);
    std::vector<double> joint_values = { -1.0, 0.7, 0.7, -1.5, -0.7, 2.0, 0.0 };
    goal_state.setJointGroupPositions(joint_model_group, joint_values);
    moveit_msgs::Constraints joint_goal = kinematic_constraints::constructGoalConstraints(goal_state, joint_model_group);
    req.goal_constraints.clear();
    req.goal_constraints.push_back(joint_goal);
  
    // Before planning, we will need a Read Only lock on the planning scene so that it does not modify the world representation while planning
    {
      planning_scene_monitor::LockedPlanningSceneRO lscene(psm);
      // Now, call the pipeline and check whether planning was successful.
      planning_pipeline->generatePlan(lscene, req, res);
    }
    // Check that the planning was successful
    if (res.error_code_.val != res.error_code_.SUCCESS)
    {
      ROS_ERROR("Could not compute plan successfully");
      return 0;
    }
    / /Visualize the trajectory
    ROS_INFO("Visualizing the trajectory");
    res.getMessage(response);
    display_trajectory.trajectory_start = response.trajectory_start;
    display_trajectory.trajectory.push_back(response.trajectory);
    // Now you should see two planned trajectories in series
    display_publisher.publish(display_trajectory);
    visual_tools.publishTrajectoryLine(display_trajectory.trajectory.back(), joint_model_group);
    visual_tools.trigger();
    // Wait for user input
    visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to continue the demo");
  
      
    //********************** Using a Planning Request Adapter **********************
    // A planning request adapter allows us to specify a series of operations that should happen either before planning takes place or after the planning has been done on the resultant path合成路径
  
    // First, set the state in the planning scene to the final state of the last plan
    robot_state = planning_scene_monitor::LockedPlanningSceneRO(psm)->getCurrentStateUpdated(response.trajectory_start);
    robot_state->setJointGroupPositions(joint_model_group, response.trajectory.joint_trajectory.points.back().positions);
    moveit::core::robotStateToRobotStateMsg(*robot_state, req.start_state);
    // Now, set one of the joints slightly稍微 outside its upper limit
    const moveit::core::JointModel* joint_model = joint_model_group->getJointModel("panda_joint3");
    const moveit::core::JointModel::Bounds& joint_bounds = joint_model->getVariableBounds();
    std::vector<double> tmp_values(1, 0.0);
    tmp_values[0] = joint_bounds[0].min_position_ - 0.01;
    robot_state->setJointPositions(joint_model, tmp_values);
    req.goal_constraints.clear();
    req.goal_constraints.push_back(pose_goal);
  
    // Before planning, we will need a Read Only lock on the planning scene so that it does not modify the world representation while planning
    {
      planning_scene_monitor::LockedPlanningSceneRO lscene(psm);
      // Now, call the pipeline and check whether planning was successful.
      planning_pipeline->generatePlan(lscene, req, res);
    }
    if (res.error_code_.val != res.error_code_.SUCCESS)
    {
      ROS_ERROR("Could not compute plan successfully");
      return 0;
    }
    // Visualize the trajectory
    ROS_INFO("Visualizing the trajectory");
    res.getMessage(response);
    display_trajectory.trajectory_start = response.trajectory_start;
    display_trajectory.trajectory.push_back(response.trajectory);
    // Now you should see three planned trajectories in series
    display_publisher.publish(display_trajectory);
    visual_tools.publishTrajectoryLine(display_trajectory.trajectory.back(), joint_model_group);
    visual_tools.trigger();
  
    // Wait for user input 
    visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to finish the demo");
  
    ROS_INFO("Done");
    return 0;
  }
  ```

  

# 七、MoveIt Plugins

## 7.1 Plugin

- [ROS pluginlib](http://wiki.ros.org/pluginlib)的添加教程

  两个必要的类：

  1. Base class
  2. plugin class:从base class继承而来

- 检查plugin

  ```
  rospack plugins --attrib=plugin moveit_core
  ```

- 如何将一个新的motion planner作为plugin添加到MoveIt

  - MoveIt中的Base class是`planning_interface`，任何新的planner plugin都要继承他。

  - 接下创建一个plugin：linear interpolation planner(lerp)。它会规划joint space中的2个state之间的motion。

  - 在moveit中添加新的planner的类之间的关系

    ![](https://github.com/Fernweh-yang/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/ROS/lerp_planner.png?raw=true)

## 7.2 创建一个plugin的[代码](https://github.com/ros-planning/moveit_tutorials/tree/master/doc/creating_moveit_plugins/lerp_motion_planner/src)

1. 在src中写`lerp_planner_manager.cpp`来override重写基类`planning_interface`的`PlnnerManager`类

   - 在代码的最后通过`class_loader`的`CLASS_LOADER_REGISTER_CLASS`macro宏来将`LERPPlanPlannerManager`register注册为一个plugin

     ```c++
     CLASS_LOADER_REGISTER_CLASS(emptyplan_interface::EmptyPlanPlannerManager, planning_interface::PlannerManager);
     ```

2. 在src中写`lerp_planning_context.cpp`类来覆写积累`planning_interface`的`PlanningContext`类

   - 这个类是用来定义solve method的

   - 这里的响应函数response function定义在`moveit_msgs::MotionPlanDetailedResponse`

   - `PlannerConfigurationSettings`可以用于传递planner-specific的参数

   - 使用ROS param server从xx.yaml中也可以传递planner-specific的参数
     - 代码使用`lerp_planning.yaml` in `panda_moveit_config` package

3. 使用一个xml文件`emptyplan_interface_plugin_description.xml`描述description

   ```xml
   <library  path="libmoveit_emptyplan_planner_plugin">
     <class name="emptyplan_interface/EmptyPlanPlanner" type="emptyplan_interface::EmptyPlanPlannerManager" base_class_type="planning_interface::PlannerManager">
      <description>
      </description>
     </class>	
   </library>
   ```

4. 在package.xml中添加如下代码将上面的xml文件p输出export到ROS Toolchain

   ```xml
   <export>
      <moveit_core plugin="${prefix}/emptyplan_interface_plugin_description.xml"/>
   </export
   ```

## 7.3 Plugin的使用

- src文件中的[lerp_example](https://github.com/ros-planning/moveit_tutorials/blob/master/doc/creating_moveit_plugins/lerp_motion_planner/src/lerp_example.cpp)创建了一个node来使用上面的plugin

## 7.4 Controller Manager Plugin

- MoveIt controller managers, somewhat a misnomer多少有点词不达意, are the interfaces to your custom自定义 low level controllers. A better way to think of them are *controller interfaces*. For most use cases, the included [MoveItSimpleControllerManager](https://github.com/ros-planning/moveit/blob/master/moveit_plugins/moveit_simple_controller_manager) is sufficient足够 if your robot controllers already provide ROS actions for FollowJointTrajectory. If you use *ros_control*, the included [MoveItRosControlInterface](https://github.com/ros-planning/moveit/blob/master/moveit_plugins/moveit_ros_control_interface) is also ideal.

  However, for some applications you might desire a more custom controller manager. An example template for starting your custom controller manager is provided [here](https://github.com/ros-planning/moveit_tutorials/blob/master/doc/controller_configuration/src/moveit_controller_manager_example.cpp).

## 7.5 Example Constraint Sampler Plugin

- Create a `ROBOT_moveit_plugins` package and within that a sub-folder for your `ROBOT_constraint_sampler` plugin. Modify the template provided by `ROBOT_moveit_plugins/ROBOT_moveit_constraint_sampler_plugin`

- In your `ROBOT_moveit_config/launch/move_group.launch` file, within the `<node name="move_group">`, add the parameter:

  ```
  <param name="constraint_samplers" value="ROBOT_moveit_constraint_sampler/ROBOTConstraintSamplerAllocator"/>
  ```

- Now when you launch move_group, it should default to your new constraint sampler.

# 八、Visualizing Collision

- 运行：roslaunch moveit_tutorials visualizing_collisions_tutorial.launch
- 主要使用[InteractiveRobot](http://docs.ros.org/en/groovy/api/pr2_moveit_tutorials/html/classInteractiveRobot.html)类

# 九、Time Parameterization

- MoveIt 目前主要是一个运动学kinematik运动规划框架framework——它规划关节或末端执行器的位置，但不规划速度或加速度。 然而，MoveIt 确实利用utilize后处理post process来对速度和加速度值的运动轨迹进行时间参数化。

- 通常moveit设置的速度加速度来自robot’s URDF or `joint_limits.yaml`.

  - Specific joint properties can be changed with the keys `max_position, min_position, max_velocity, max_acceleration`. 
  - Joint limits can be turned on or off with the keys `has_velocity_limits`和` has_acceleration_limits`.

- 在运行时runtime,也可以设置速度（最大速度的几分之几0-1）

  -  you can set the two scaling factors as described in [MotionPlanRequest.msg](http://docs.ros.org/noetic/api//moveit_msgs/html/msg/MotionPlanRequest.html). 
  - 也可以在[MoveIt MotionPlanning RViz plugin](https://ros-planning.github.io/moveit_tutorials/doc/quickstart_in_rviz/quickstart_in_rviz_tutorial.html)rviz中设置

- 三种时间参数化算法：

  - [Iterative Parabolic Time Parameterization](https://github.com/ros-planning/moveit/blob/master/moveit_core/trajectory_processing/src/iterative_time_parameterization.cpp)

    - The Iterative Parabolic Time Parameterization algorithm is used by default in the [Motion Planning Pipeline](https://ros-planning.github.io/moveit_tutorials/doc/motion_planning_pipeline/motion_planning_pipeline_tutorial.html) as a Planning Request Adapter as documented in [this tutorial](https://ros-planning.github.io/moveit_tutorials/doc/motion_planning_pipeline/motion_planning_pipeline_tutorial.html#using-a-planning-request-adapter). Although the Iterative Parabolic Time Parameterization algorithm MoveIt uses has been used by hundreds of robots over the years, there is known [bug with it](https://github.com/ros-planning/moveit/issues/160).

  - [Iterative Spline Parameterization](https://github.com/ros-planning/moveit/blob/master/moveit_core/trajectory_processing/src/iterative_spline_parameterization.cpp)

    - The Iterative Spline Parameterization algorithm was merged with [PR 382](https://github.com/ros-planning/moveit/pull/382) as an approach to deal with these issues. While preliminary experiments are very promising, we are waiting for more feedback from the community before replacing the Iterative Parabolic Time Parameterization algorithm completely.

  - [Time-optimal Trajectory Generation](https://github.com/ros-planning/moveit/blob/master/moveit_core/trajectory_processing/src/time_optimal_trajectory_generation.cpp)

    - Time-optimal Trajectory Generation introduced in PRs [#809](https://github.com/ros-planning/moveit/pull/809) and [#1365](https://github.com/ros-planning/moveit/pull/1365) produces trajectories with very smooth and continuous velocity profiles. The method is based on fitting path segments to the original trajectory and then sampling new waypoints from the optimized path. This is different from strict time parameterization methods as resulting waypoints may divert from the original trajectory within a certain tolerance. As a consequence, additional collision checks might be required when using this method.

      Open Source Feedback

# 十、Planning with Approximated Constraint Manifolds

近似约束流行的规划，即设置一个constraint Database来约束规划路径，见[教程](https://ros-planning.github.io/moveit_tutorials/doc/planning_with_approximated_constraint_manifolds/planning_with_approximated_constraint_manifolds_tutorial.html)

# 十一、Pick and Place

- 2个shell
  - roslaunch panda_moveit_config demo.launch
  - rosrun moveit_tutorials pick_place_tutorial
- [教程](https://ros-planning.github.io/moveit_tutorials/doc/pick_place/pick_place_tutorial.html)
- [代码](https://github.com/ros-planning/moveit_tutorials/tree/master/doc/pick_place)



# 十二、URDF，SRDF和 Xacro

## 12.1 URDF

- 一些学习网站

  - [URDF ROS Wiki Page](http://www.ros.org/wiki/urdf) - The URDF ROS Wiki page is the source of most information about the URDF.

  - [URDF Tutorials](http://www.ros.org/wiki/urdf/Tutorials) - Tutorials for working with the URDF.

  - [SOLIDWORKS URDF Plugin](http://www.ros.org/wiki/sw_urdf_exporter) - A plugin that lets you generate a URDF directly from a SOLIDWORKS model.

  - [URDF Examples](https://wiki.ros.org/urdf/Examples)

### 12.1.0如何rviz显示urdf
- [展示一个urdf](https://blog.csdn.net/xuehuafeiwu123/article/details/60764997)(urdf_tutorial)

  ```shell
  # 输入绝对路径
  $ roslaunch urdf_tutorial display.launch model:=/opt/ros/noetic/share/urdf_tutorial/urdf/01-myfirst.urdf
  # 也可以直接寻找，不管shell在什么路径下使用都可以
  $ roslaunch urdf_tutorial display.launch model:='$(find urdf_tutorial)/urdf/01-myfirst.urdf'
  ```

  - 用到的launch file:

    至少需要包含3个节点

  ```xml
  <launch>
      <arg name="model" default="$(find urdf_tutorial)/urdf/01-myfirst.urdf"/>
      <!--专门在 rviz 中使用的。可以显示各关节的滑动条。-->
      <arg name="gui" default="true" />
      <arg name="rvizconfig" default="$(find urdf_tutorial)/rviz/urdf.rviz" />
  	
      <!--第一个：加载urdf模型-->
      <param name="robot_description" command="$(find xacro)/xacro $(arg model)" />
      
      <!--第二个：joint_ state_publisher用于读取机器人模型中的参数，并发布一系列的变换矩阵组成机器人的 tf 树。-->
      <node if="$(arg gui)" name="joint_state_publisher" pkg="joint_state_publisher_gui" type="joint_state_publisher_gui" />
      <node unless="$(arg gui)" name="joint_state_publisher" pkg="joint_state_publisher" type="joint_state_publisher" />
      
      <!--第三个：robot_ state_publisher 发出机器人的状态-->>
      <node name="robot_state_publisher" pkg="robot_state_publisher" type="robot_state_publisher" />
      
      <!--打开rviz设置文件-->
      <node name="rviz" pkg="rviz" type="rviz" args="-d $(arg rvizconfig)" required="true" />
  
  </launch>
  ```

### 12.1.1Fixed joint Robot

- 一个只有固定joint的机器人

  - mesh的使用见代码180行

    mesh文件一般分2种：

    - <.dae> 格式的，主要用来显示模型
    -  <.stl > 格式的,主要用来进行碰撞检测。

    可以用各种三维制图软件来生成。如果不进行碰撞检测，完全可以用圆柱体、长方体等常见形状来代替。只要质量、质心、惯量矩阵等设置正确，运动学和动力学仿真没有任何问题。显示模型的形状只是为了好看，而其设置的属性才是根本。

  ```xml
  <?xml version="1.0"?>
  <!--机器人的名字时visual-->
  <robot name="visual">
      <!--定义三种不同的颜色-->
      <!--可以添加texture纹理，用一张图来作物体的表面-->
      <material name="blue">
          <color rgba="0 0 0.8 1"/>
      </material>
  
      <material name="black">
          <color rgba="0 0 0 1"/>
      </material>
  
      <material name="white">
          <color rgba="1 1 1 1"/>
      </material>
  
      <!--定义一个名为base_link的link-->
      <link name="base_link">
          <!--下面是可以看到的部分-->
          <visual>
              <!--几何性质-->
              <geometry>
                  <!--一个长0.6米，半径为0.2米的圆柱-->
                  <cylinder length="0.6" radius="0.2"/>
              </geometry>
              <!--用到上面自定义的材料-->
              <material name="blue"/>
          </visual>
      </link>
  
      <link name="right_leg">
          <visual>
              <geometry>
                  <!--这里的几何是一个长宽高为如下的盒子-->
                  <box size="0.6 0.1 0.2"/>
              </geometry>
              <!--相对于自己的origin原点的位移和旋转-->
              <!--rpy表示roll,pitch,yaw-->
              <origin rpy="0 1.57075 0" xyz="0 0 -0.3"/>
              <material name="white"/>
          </visual>
      </link>
      <!--连接2个link需要用到joint,fixed表示这是个不能动的关节-->
      <joint name="base_to_right_leg" type="fixed">
          <parent link="base_link"/>
          <child link="right_leg"/>
          <!--表示child link的原点origin相对于parent link的origin偏移了xyz-->
          <origin xyz="0 -0.22 0.25"/>
      </joint>
  
      <link name="right_base">
          <visual>
              <geometry>
                  <box size="0.4 0.1 0.1"/>
              </geometry>
              <material name="white"/>
          </visual>
      </link>
      <joint name="right_base_joint" type="fixed">
          <parent link="right_leg"/>
          <child link="right_base"/>
          <origin xyz="0 0 -0.6"/>
      </joint>
  
      <link name="right_front_wheel">
          <visual>
              <origin rpy="1.57075 0 0" xyz="0 0 0"/>
              <geometry>
                  <cylinder length="0.1" radius="0.035"/>
              </geometry>
              <material name="black"/>
              <origin rpy="0 0 0" xyz="0 0 0"/>
          </visual>
      </link>
      <joint name="right_front_wheel_joint" type="fixed">
          <parent link="right_base"/>
          <child link="right_front_wheel"/>
          <origin rpy="0 0 0" xyz="0.133333333333 0 -0.085"/>
      </joint>
  
      <link name="right_back_wheel">
          <visual>
              <origin rpy="1.57075 0 0" xyz="0 0 0"/>
              <geometry>
                  <cylinder length="0.1" radius="0.035"/>
              </geometry>
              <material name="black"/>
          </visual>
      </link>
      <joint name="right_back_wheel_joint" type="fixed">
          <parent link="right_base"/>
          <child link="right_back_wheel"/>
          <origin rpy="0 0 0" xyz="-0.133333333333 0 -0.085"/>
      </joint>
  
      <link name="left_leg">
          <visual>
              <geometry>
                  <box size="0.6 0.1 0.2"/>
              </geometry>
              <origin rpy="0 1.57075 0" xyz="0 0 -0.3"/>
              <material name="white"/>
          </visual>
      </link>
      <joint name="base_to_left_leg" type="fixed">
          <parent link="base_link"/>
          <child link="left_leg"/>
          <origin xyz="0 0.22 0.25"/>
      </joint>
  
      <link name="left_base">
          <visual>
              <geometry>
                  <box size="0.4 0.1 0.1"/>
              </geometry>
              <material name="white"/>
          </visual>
      </link>
      <joint name="left_base_joint" type="fixed">
          <parent link="left_leg"/>
          <child link="left_base"/>
          <origin xyz="0 0 -0.6"/>
      </joint>
  
      <link name="left_front_wheel">
          <visual>
              <origin rpy="1.57075 0 0" xyz="0 0 0"/>
              <geometry>
                  <cylinder length="0.1" radius="0.035"/>
              </geometry>
              <material name="black"/>
          </visual>
      </link>
      <joint name="left_front_wheel_joint" type="fixed">
          <parent link="left_base"/>
          <child link="left_front_wheel"/>
          <origin rpy="0 0 0" xyz="0.133333333333 0 -0.085"/>
      </joint>
  
      <link name="left_back_wheel">
          <visual>
              <origin rpy="1.57075 0 0" xyz="0 0 0"/>
              <geometry>
                  <cylinder length="0.1" radius="0.035"/>
              </geometry>
              <material name="black"/>
          </visual>
      </link>
      <joint name="left_back_wheel_joint" type="fixed">
          <parent link="left_base"/>
          <child link="left_back_wheel"/>
          <origin rpy="0 0 0" xyz="-0.133333333333 0 -0.085"/>
      </joint>
  
      <joint name="gripper_extension" type="fixed">
          <parent link="base_link"/>
          <child link="gripper_pole"/>
          <origin rpy="0 0 0" xyz="0.19 0 0.2"/>
      </joint>
      <link name="gripper_pole">
          <visual>
              <geometry>
                  <cylinder length="0.2" radius="0.01"/>
              </geometry>
              <origin rpy="0 1.57075 0 " xyz="0.1 0 0"/>
          </visual>
      </link>
  
      <joint name="left_gripper_joint" type="fixed">
          <origin rpy="0 0 0" xyz="0.2 0.01 0"/>
          <parent link="gripper_pole"/>
          <child link="left_gripper"/>
      </joint>
  
      <link name="left_gripper">
          <visual>
              <origin rpy="0.0 0 0" xyz="0 0 0"/>
              <geometry>
                  <!--通过mesh来秒速一些复杂图形-->
                  <!--通过package://NAME_OF_PACKAGE/path notation来调用-->
                  <mesh filename="package://urdf_tutorial/meshes/l_finger.dae"/>
              </geometry>
          </visual>
      </link>
  
      <joint name="left_tip_joint" type="fixed">
          <parent link="left_gripper"/>
          <child link="left_tip"/>
      </joint>
      <link name="left_tip">
          <visual>
              <origin rpy="0.0 0 0" xyz="0.09137 0.00495 0"/>
              <geometry>
                  <mesh filename="package://urdf_tutorial/meshes/l_finger_tip.dae"/>
              </geometry>
          </visual>
      </link>
      <joint name="right_gripper_joint" type="fixed">
          <origin rpy="0 0 0" xyz="0.2 -0.01 0"/>
          <parent link="gripper_pole"/>
          <child link="right_gripper"/>
      </joint>
  
      <link name="right_gripper">
          <visual>
              <origin rpy="-3.1415 0 0" xyz="0 0 0"/>
              <geometry>
                  <mesh filename="package://urdf_tutorial/meshes/l_finger.dae"/>
              </geometry>
          </visual>
      </link>
      <joint name="right_tip_joint" type="fixed">
          <parent link="right_gripper"/>
          <child link="right_tip"/>
      </joint>
  
      <link name="right_tip">
          <visual>
              <origin rpy="-3.1415 0 0" xyz="0.09137 0.00495 0"/>
              <geometry>
                  <mesh filename="package://urdf_tutorial/meshes/l_finger_tip.dae"/>
              </geometry>
          </visual>
      </link>
  
      <link name="head">
          <visual>
              <geometry>
              <!--这里的几何图形是一个圆球-->
              <sphere radius="0.2"/>
                  </geometry>
              <material name="white"/>
          </visual>
      </link>
      <joint name="head_swivel" type="fixed">
          <parent link="base_link"/>
          <child link="head"/>
          <origin xyz="0 0 0.3"/>
      </joint>
  
      <link name="box">
          <visual>
              <geometry>
                  <box size="0.08 0.08 0.08"/>
              </geometry>
              <material name="blue"/>
          </visual>
      </link>
      <joint name="tobox" type="fixed">
          <parent link="head"/>
          <child link="box"/>
          <origin xyz="0.1814 0 0.1414"/>
      </joint>
  </robot>
  ```

### 12.1.2 Flexible joint Robot

12.1.1中所有的joint都是固定不能动的,现在要revise修改为可动的。

- 修改头和身体之间的joint，轮子与脚之间的joint的类型为**continuous**

  ```xml
  <!--头和身体-->
  <!--continuous可以让两个link饶某个轴随意的旋转-->
  <joint name="head_swivel" type="continuous">
      <parent link="base_link"/>
      <child link="head"/>
      <!--定义旋转轴为z轴-->
      <axis xyz="0 0 1"/>
      <origin xyz="0 0 0.3"/>
  </joint>
  <!--有四个轮子，下面为其中一个和他的脚-->
  <joint name="left_back_wheel_joint" type="continuous">
      <!--定义旋转轴为y轴-->
      <axis rpy="0 0 0" xyz="0 1 0"/>
      <parent link="left_base"/>
      <child link="left_back_wheel"/>
      <origin rpy="0 0 0" xyz="-0.133333333333 0 -0.085"/>
  </joint>
  ```

- 修改Gripper夹子的joint类型为**revolute**

  ```xml
  <!--有2个夹子，一左一右，都需要改成revolute-->
  <!--revolute可以像continuous一样饶轴旋转，但他们有严格strict的限制，需要limit标签-->
  <joint name="left_gripper_joint" type="revolute">
      <!--定义旋转轴为z轴-->
      <axis xyz="0 0 1"/>
      <!--定义上下限(radians弧度制)，定义最大速度和力-->
      <limit effort="1000.0" lower="0.0" upper="0.548" velocity="0.5"/>
      <origin rpy="0 0 0" xyz="0.2 0.01 0"/>
      <parent link="gripper_pole"/>
      <child link="left_gripper"/>
  </joint>
  ```

- 修改Gripper夹子的手臂的joint类型为**prismatic**

  ```xml
  <!--prismatic可以沿着某个轴运动-->
  <joint name="gripper_extension" type="prismatic">
      <parent link="base_link"/>
      <child link="gripper_pole"/>
      <!--定义上下限(meter米)，定义最大速度和力-->
      <limit effort="1000.0" lower="-0.38" upper="0" velocity="0.5"/>
      <origin rpy="0 0 0" xyz="0.19 0 0.2"/>
  </joint>
  ```

- 其他类型的joint

  - **planar joint**：相对于prismatic只能沿着一个轴的移动，planar可以在一个平面中四处移动
  - **floating joint**：可以在三维中任意的移动

- Specify指定 the pose

  当我们在rviz中拖动joint的控制滚条，会进行如下操作

  1. [joint_state_publisher](https://wiki.ros.org/joint_state_publisher)会parse语法分析URDF并找到所有的可移动joint和他们的限制limit
  2. 接着joint_state_publisher会将控制滚条的值以[sensor_msgs/JointState](http://docs.ros.org/en/api/sensor_msgs/html/msg/JointState.html)msg发送
  3. 这个msg会被[robot_state_publisher](https://wiki.ros.org/robot_state_publisher)用来计算不同部分之间的移动transforms.
  4. 最后生成的变化树resulting transform tree被用来在rviz中显示所有的形状shapes。

### 12.1.3 Physical and Collision Properties

- [Here is the new urdf](https://raw.githubusercontent.com/ros/urdf_tutorial/master/urdf/07-physics.urdf) with collision and physical properties.

- Collision:

  用于碰撞检测

  ```xml
  <link name="base_link">
      <visual>
          <geometry>
          	<cylinder length="0.6" radius="0.2"/>
          </geometry>
          <material name="blue"/>
      </visual>
      <collision>
          <geometry>
          	<cylinder length="0.6" radius="0.2"/>
          </geometry>
      </collision>
  </link>
  ```

  - collison和visual同级，是link的直接subelement
  - 和visual一样，设置碰撞检测的形状
  - 虽然现实中碰撞和可见图形是一致的，但在如下两种场景，应该用更简单的碰撞几何来替代：
    - Quicker Processing：因为碰撞计算更复杂，所以如果需要节省计算资源，可以使用简单的geometries几何来代替复杂的collision elements
    - Safe Zones：有些sensitive equipment敏感部件，我们不希望任何东西与他碰撞。比如机器人的头部，我们就可以把头的碰撞几何设置为一个包围它的圆柱，以防任何东西太靠近他的头部。

- Physical Properties

  为了正确的properly模拟，需要link有相关的物理性质

  ```xml
  <link name="base_link">
      <visual>
          <geometry>
          	<cylinder length="0.6" radius="0.2"/>
          </geometry>
          <material name="blue"/>
      </visual>
      <collision>
          <geometry>
          	<cylinder length="0.6" radius="0.2"/>
          </geometry>
      </collision>
      <inertial>
          <!--质量单位是kilograms-->
          <mass value="10"/>
          <!--rotational inertia matrix旋转惯量矩阵，是一个symmetrical对称矩阵-->
          <!--如果不确定，可以用ixx/iyy/izz=1e-3 or smaller作为默认值，这是对于一个中型尺寸的link而言的(比如一个0.1m长，0.6kg的盒子)-->
          <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
      </inertial>
  </link>  
  
  ```

  - inertial和collision,visual一样，都是link的直接subelement

  - Contact Coefficients

    - mu - [Friction coefficient](https://simple.wikipedia.org/wiki/Coefficient_of_friction)摩擦系数
    - kp - [Stiffness coefficient](https://en.wikipedia.org/wiki/Stiffness)弹性系数
    - kd - [Dampening coefficient](https://en.wikipedia.org/wiki/Damping_ratio#Definition)阻尼系数

  - Joint Dynamics

    关节移动由joint的dynamics tag定义，有2个attribute属性：

    1. friction - The physical static friction静态摩擦. For prismatic平移关节 joints, the units are Newtons. For 旋转关节revolving joints, the units are Newton meters.
    2. damping - The physical damping value阻尼值. For prismatic joints, the units are Newton seconds per meter. For revolving joints, Newton meter seconds per radian

  - Jonit 的其他一些[tag](https://wiki.ros.org/urdf/XML/joint)

    - safety_controller(optional)
    - mimic(optional)
    - dynamics(optional)
    - calibration(optional)

## 12.2 SRDF

- SRDF是对URDF的补充

## 12.3 [MoveIt Setup Assistant](https://ros-planning.github.io/moveit_tutorials/doc/setup_assistant/setup_assistant_tutorial.html)

- 用于生成SRDF

## 12.4 [Xacro](http://wiki.ros.org/xacro)

[Xacro](https://wiki.ros.org/xacro)名字由来： macro宏 language for XML

- 相比urdf提供了如下三种特性，帮忙降低模型开发难度并降低了模型描述的复杂度

  - Constants常值

  - Simple Math数学计算

  - Macros宏

### 12.4.1 Using Xacro

- 通常的使用如下：

  写好Xacro后，先将xacro转化为urdf，再使用

​		`xacro --inorder model.xacro > model.urdf `

- 也可以直接在launch file中自动生成urdf，但这会花费更多的时间来启动节点

    ```xml
    <param name="robot_description" command="xacro --inorder '$(find pr2_description)/robots/pr2.urdf.xacro'" />
    ```

- 在xml文件的开头需要

    ```xml
    <?xml version="1.0"?>
    <!--根标签：必须指明xmlns:xacro-->
    <robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="firefighter">
        ...
    </robot>
    ```

### 12.4.2 Constants

```xml
<!--使用xacro的语法必须要用xacro:，其他的都和urdf一样-->
<xacro:property name="width" value="0.2" />
<xacro:property name="bodylen" value="0.6" />
<link name="base_link">
    <visual>
        <geometry>
            <cylinder radius="${width}" length="${bodylen}"/>
        </geometry>
        <material name="blue"/>
    </visual>
    <collision>
        <geometry>
            <cylinder radius="${width}" length="${bodylen}"/>
        </geometry>
    </collision>
</link>
```

- 相比于urdf，xacro可以设置常值，这样一些不同部件但相同的固定的参数如长度，宽度就可以统一设置成一个constant,之后要调整也很方便。

- 我们还能自动组合constants

  ```xml
  <xacro:property name=”robotname” value=”marvin” />
  <link name=”${robotname}s_leg” />
  <!--上面2行等价于下面-->
  <link name=”marvins_leg” />
  ```

### 12.4.3 Math

xacaro也支持简单的数学计算,

```xml
<cylinder radius="${wheeldiam/2}" length="0.1"/>
<origin xyz="${reflect*(width+.02)} 0 0.25" />
```

- 所有的数学计算，数据类型都是floats

### 12.4.4 Macros

#### 1. Simple Macro

```xml
<!--定义宏结构-->
<xacro:macro name="default_origin">
    <origin xyz="0 0 0" rpy="0 0 0"/>
</xacro:macro>
<!--调用定义过的宏-->
<xacro:default_origin />
<!--上面6行等价于-->
<origin rpy="0 0 0" xyz="0 0 0"/>
```

#### 2. Parameterized Macro

- 把一个参数传入xacro macro 中

  ```xml
  <!--定义宏结构-->
  <xacro:macro name="default_inertial" params="mass">
      <inertial>
              <mass value="${mass}" />
              <inertia ixx="1.0" ixy="0.0" ixz="0.0"
                   iyy="1.0" iyz="0.0"
                   izz="1.0" />
      </inertial>
  </xacro:macro>
  <!--调用定义过的宏-->
  <xacro:default_inertial mass="10"/>
  ```

- 也可以传入一个block parameter

  ```xml
  <!--定义宏结构-->
  <!--如果是一个block parameter，需要再parameter名字前加上*asterisk号-->
  <xacro:macro name="blue_shape" params="name *shape">
      <link name="${name}">
          <visual>
              <geometry>
                  <!--使用xacro:insert_block来插入-->
                  <xacro:insert_block name="shape" />
              </geometry>
              <material name="blue"/>
          </visual>
          <collision>
              <geometry>
                  <xacro:insert_block name="shape" />
              </geometry>
          </collision>
      </link>
  </xacro:macro>
  <!--调用定义过的宏,并给与块结构-->
  <xacro:blue_shape name="base_link">
      <cylinder radius=".42" length=".01" />
  </xacro:blue_shape>
  ```

#### 3. 实际使用:Leg Macro

```xml
<xacro:macro name="leg" params="prefix reflect">
    <link name="${prefix}_leg">
        <visual>
            <geometry>
                <box size="${leglen} 0.1 0.2"/>
            </geometry>
            <origin xyz="0 0 -${leglen/2}" rpy="0 ${pi/2} 0"/>
            <material name="white"/>
        </visual>
        <collision>
            <geometry>
                <box size="${leglen} 0.1 0.2"/>
            </geometry>
            <origin xyz="0 0 -${leglen/2}" rpy="0 ${pi/2} 0"/>
        </collision>
        <xacro:default_inertial mass="10"/>
    </link>

    <joint name="base_to_${prefix}_leg" type="fixed">
        <parent link="base_link"/>
        <child link="${prefix}_leg"/>
        <origin xyz="0 ${reflect*(width+.02)} 0.25" />
    </joint>
    <!-- A bunch of stuff cut -->
</xacro:macro>
<!--定义一遍宏，创建2个腿-->
<!--第一个参数prefix定义了创建的是哪条腿-->
<!--第二个参数reflect用于计算origin，即腿的位置-->
<xacro:leg prefix="right" reflect="1" />
<xacro:leg prefix="left" reflect="-1" />
```

## 12.5 Using URDF in Gazebo

[代码例子](https://github.com/ros/urdf_sim_tutorial)

### 12.5.1 launch file

运行：`roslaunch urdf_sim_tutorial 13-diffdrive.launch`

一共有2个launch file:

1. 13-diffdrive.launch

    ```xml
    <launch>
        <!--读取模型-->
        <arg name="model" default="$(find urdf_sim_tutorial)/urdf/13-diffdrive.urdf.xacro"/>
        <!--加载rviz配置文件-->
        <arg name="rvizconfig" default="$(find urdf_tutorial)/rviz/urdf.rviz" />
        <!--启动另一个launch file,用于启动gazebo-->
        <include file="$(find urdf_sim_tutorial)/launch/gazebo.launch">
        <arg name="model" value="$(arg model)" />
        </include>
        <!--按上面加载的rviz配置来打开rviz-->
        <node name="rviz" pkg="rviz" type="rviz" args="-d $(arg rvizconfig)" />
        <!--读取四个yaml.分别用于不同的控制器(ns:命名空间)-->
        <!--yaml里存放的参数会传递给ROS parameter space然后发送给gazebo进行控制-->
        <rosparam command="load"
                file="$(find urdf_sim_tutorial)/config/joints.yaml"
                ns="r2d2_joint_state_controller" />
        <rosparam command="load"
                file="$(find urdf_sim_tutorial)/config/head.yaml"
                ns="r2d2_head_controller" />
        <rosparam command="load"
                file="$(find urdf_sim_tutorial)/config/gripper.yaml"
                ns="r2d2_gripper_controller" />
        <rosparam command="load"
                file="$(find urdf_sim_tutorial)/config/diffdrive.yaml"
                ns="r2d2_diff_drive_controller" />
        
    	<!--使用controller_manage将上面4个命名空间(ns)传入gazebo-->
        <node name="r2d2_controller_spawner" pkg="controller_manager" type="spawner"
        args="r2d2_joint_state_controller
              r2d2_head_controller
              r2d2_gripper_controller
              r2d2_diff_drive_controller
              --shutdown-timeout 3"/>
    	<!--rqt_robot_steering是rqt_robot_plugins提供的包-->
        <!--rqt_robot_steering提供了一个GUI Plugin用于驾驶机器人-->
        <node name="rqt_robot_steering" pkg="rqt_robot_steering" type="rqt_robot_steering">
            <!--rqt_robot_steering发送的topic-->
        	<param name="default_topic" value="/r2d2_diff_drive_controller/cmd_vel"/>
        </node>
    </launch>
    ```

2. gazobo.launch

   ```xml
   <launch>
   
       <!-- these are the arguments you can pass this launch file, for example paused:=true -->
       <arg name="paused" default="false"/>
       <arg name="use_sim_time" default="true"/>
       <arg name="gui" default="true"/>
       <arg name="headless" default="false"/>
       <arg name="debug" default="false"/>
       <arg name="model" default="$(find urdf_tutorial)/urdf/08-macroed.urdf.xacro"/>
   
       <!-- We resume the logic in empty_world.launch, changing only the name of the world to be launched -->
       <include file="$(find gazebo_ros)/launch/empty_world.launch">
           <arg name="debug" value="$(arg debug)" />
           <arg name="gui" value="$(arg gui)" />
           <arg name="paused" value="$(arg paused)"/>
           <arg name="use_sim_time" value="$(arg use_sim_time)"/>
           <arg name="headless" value="$(arg headless)"/>
       </include>
       
   	<!--读取模型-->
       <param name="robot_description" command="$(find xacro)/xacro $(arg model)" />
   
       <!-- push robot_description to factory and spawn robot in gazebo -->
       <node name="urdf_spawner" pkg="gazebo_ros" type="spawn_model"
       args="-z 1.0 -unpause -urdf -model robot -param robot_description" respawn="false" output="screen" />
   	<!--发送机器人状态-->
       <node pkg="robot_state_publisher" type="robot_state_publisher"  name="robot_state_publisher">
       	<param name="publish_frequency" type="double" value="30.0" />
       </node>
   
   </launch>
   ```
   
### 12.5.2 yaml file

第一个type:"xx"，表明用的是哪个控制器。具体见[ros_controllers](http://wiki.ros.org/ros_controllers?distro=noetic)和[ros_control](http://wiki.ros.org/ros_control?distro=noetic)

   1. joints.yaml
   
      ```yaml
      # The joint state controller handles publishing transforms for any moving joints
      type: "joint_state_controller/JointStateController"
      publish_rate: 50
      ```
   
   2. head.yaml
   
      ```yaml
      type: "position_controllers/JointPositionController"
      joint: head_swivel
      ```
   
   3. gripper.yaml
   
      ```yaml
      type: "position_controllers/JointGroupPositionController"
      joints:
       - gripper_extension
       - left_gripper_joint
       - right_gripper_joint
      ```
   
   4. diffdrive.yam
   
      ```yaml
      type: "diff_drive_controller/DiffDriveController"
      publish_rate: 50
      
      left_wheel: ['left_front_wheel_joint', 'left_back_wheel_joint']
      right_wheel: ['right_front_wheel_joint', 'right_back_wheel_joint']
      
      wheel_separation: 0.44
      
      # Odometry covariances for the encoder output of the robot. These values should
      # be tuned to your robot's sample odometry data, but these values are a good place
      # to start
      pose_covariance_diagonal: [0.001, 0.001, 0.001, 0.001, 0.001, 0.03]
      twist_covariance_diagonal: [0.001, 0.001, 0.001, 0.001, 0.001, 0.03]
      
      # Top level frame (link) of the robot description
      base_frame_id: base_link
      
      # Velocity and acceleration limits for the robot
      linear:
        x:
          has_velocity_limits    : true
          max_velocity           : 0.2   # m/s
          has_acceleration_limits: true
          max_acceleration       : 0.6   # m/s^2
      angular:
        z:
          has_velocity_limits    : true
          max_velocity           : 2.0   # rad/s
          has_acceleration_limits: true
          max_acceleration       : 6.0   # rad/s^2
      ```

### 12.5.3 xacro

[整体的xacro](https://github.com/ros/urdf_sim_tutorial/blob/master/urdf/13-diffdrive.urdf.xacro)很长，大部分和12.1中类似，下面是和gazebo和ros下官的一些改动

- 添加Gazebo Plugin

  297-302行。用于连接gazebo和ros

  ```xml
  <gazebo>
      <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      	<robotNamespace>/</robotNamespace>
      </plugin>
  </gazebo>
  ```

- Transmission传动

  对于每一个non-fixed joint都需要设置一个transmission。他会告诉gazebo要让不同的joint做什么.

  下面每一个joint的name都必须和[整体的xacro](https://github.com/ros/urdf_sim_tutorial/blob/master/urdf/13-diffdrive.urdf.xacro)定义的joint名字一致

  1. head joint

     265-273行。控制头旋转

     ```xml
     <transmission name="head_swivel_trans">
         <type>transmission_interface/SimpleTransmission</type>
         <actuator name="$head_swivel_motor">
         	<mechanicalReduction>1</mechanicalReduction>
         </actuator>
         <!--joint名字和254行保持一致-->
         <joint name="head_swivel">
             <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
         </joint>
     </transmission>
     ```

  2. Wheel joint

     83-91行。控制轮子转动

     ```xml
     <transmission name="${prefix}_${suffix}_wheel_trans">
         <type>transmission_interface/SimpleTransmission</type>
         <actuator name="${prefix}_${suffix}_wheel_motor">
         	<mechanicalReduction>1</mechanicalReduction>
         </actuator>
         <!--joint名字和64行配对-->
         <joint name="${prefix}_${suffix}_wheel_joint">
         	<hardwareInterface>VelocityJointInterface</hardwareInterface>
         </joint>
     </transmission>
     ```

     

# 十三、Perception Pipeline

- [教程](https://ros-planning.github.io/moveit_tutorials/doc/perception_pipeline/perception_pipeline_tutorial.html)

