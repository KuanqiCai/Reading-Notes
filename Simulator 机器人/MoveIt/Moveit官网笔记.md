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

## 5.1 Planning Scene Monitor

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

## 5.2 Planning Scene ROS API

- 使用Planning Scene可以在world中添加或删除物体，固定attaching或分离detaching物体objects到机器人上

- 运行：

  - 第一个shell:`roslaunch panda_moveit_config demo.launch`
  - 第二个shell:`roslaunch moveit_tutorials planning_scene_ros_api_tutorial.launch`

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
    // BEGIN_TUTORIAL
    //
    // Visualization
    // ^^^^^^^^^^^^^
    // The package MoveItVisualTools provides many capabilities for visualizing objects, robots,
    // and trajectories in RViz as well as debugging tools such as step-by-step introspection of a script.
    moveit_visual_tools::MoveItVisualTools visual_tools("panda_link0");
    visual_tools.deleteAllMarkers();
  
    // ROS API
    // ^^^^^^^
    // The ROS API to the planning scene publisher is through a topic interface
    // using "diffs". A planning scene diff is the difference between the current
    // planning scene (maintained by the move_group node) and the new planning
    // scene desired by the user.
    //
    // Advertise the required topic
    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    // We create a publisher and wait for subscribers.
    // Note that this topic may need to be remapped in the launch file.
    ros::Publisher planning_scene_diff_publisher = node_handle.advertise<moveit_msgs::PlanningScene>("planning_scene", 1);
    ros::WallDuration sleep_t(0.5);
    while (planning_scene_diff_publisher.getNumSubscribers() < 1)
    {
      sleep_t.sleep();
    }
    visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to start the demo");
  
    // Define the attached object message
    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    // We will use this message to add or
    // subtract the object from the world
    // and to attach the object to the robot.
    moveit_msgs::AttachedCollisionObject attached_object;
    attached_object.link_name = "panda_hand";
    /* The header must contain a valid TF frame*/
    attached_object.object.header.frame_id = "panda_hand";
    /* The id of the object */
    attached_object.object.id = "box";
  
    /* A default pose */
    geometry_msgs::Pose pose;
    pose.position.z = 0.11;
    pose.orientation.w = 1.0;
  
    /* Define a box to be attached */
    shape_msgs::SolidPrimitive primitive;
    primitive.type = primitive.BOX;
    primitive.dimensions.resize(3);
    primitive.dimensions[0] = 0.075;
    primitive.dimensions[1] = 0.075;
    primitive.dimensions[2] = 0.075;
  
    attached_object.object.primitives.push_back(primitive);
    attached_object.object.primitive_poses.push_back(pose);
  
    // Note that attaching an object to the robot requires
    // the corresponding operation to be specified as an ADD operation.
    attached_object.object.operation = attached_object.object.ADD;
  
    // Since we are attaching the object to the robot hand to simulate picking up the object,
    // we want the collision checker to ignore collisions between the object and the robot hand.
    attached_object.touch_links = std::vector<std::string>{ "panda_hand", "panda_leftfinger", "panda_rightfinger" };
  
    // Add an object into the environment
    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    // Add the object into the environment by adding it to
    // the set of collision objects in the "world" part of the
    // planning scene. Note that we are using only the "object"
    // field of the attached_object message here.
    ROS_INFO("Adding the object into the world at the location of the hand.");
    moveit_msgs::PlanningScene planning_scene;
    planning_scene.world.collision_objects.push_back(attached_object.object);
    planning_scene.is_diff = true;
    planning_scene_diff_publisher.publish(planning_scene);
    visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to continue the demo");
  
    // Interlude: Synchronous vs Asynchronous updates
    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    // There are two separate mechanisms available to interact
    // with the move_group node using diffs:
    //
    // * Send a diff via a rosservice call and block until
    //   the diff is applied (synchronous update)
    // * Send a diff via a topic, continue even though the diff
    //   might not be applied yet (asynchronous update)
    //
    // While most of this tutorial uses the latter mechanism (given the long sleeps
    // inserted for visualization purposes asynchronous updates do not pose a problem),
    // it would is perfectly justified to replace the planning_scene_diff_publisher
    // by the following service client:
    ros::ServiceClient planning_scene_diff_client =
        node_handle.serviceClient<moveit_msgs::ApplyPlanningScene>("apply_planning_scene");
    planning_scene_diff_client.waitForExistence();
    // and send the diffs to the planning scene via a service call:
    moveit_msgs::ApplyPlanningScene srv;
    srv.request.scene = planning_scene;
    planning_scene_diff_client.call(srv);
    // Note that this does not continue until we are sure the diff has been applied.
    //
    // Attach an object to the robot
    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    // When the robot picks up an object from the environment, we need to
    // "attach" the object to the robot so that any component dealing with
    // the robot model knows to account for the attached object, e.g. for
    // collision checking.
    //
    // Attaching an object requires two operations
    //  * Removing the original object from the environment
    //  * Attaching the object to the robot
  
    /* First, define the REMOVE object message*/
    moveit_msgs::CollisionObject remove_object;
    remove_object.id = "box";
    remove_object.header.frame_id = "panda_hand";
    remove_object.operation = remove_object.REMOVE;
  
    // Note how we make sure that the diff message contains no other
    // attached objects or collisions objects by clearing those fields
    // first.
    /* Carry out the REMOVE + ATTACH operation */
    ROS_INFO("Attaching the object to the hand and removing it from the world.");
    planning_scene.world.collision_objects.clear();
    planning_scene.world.collision_objects.push_back(remove_object);
    planning_scene.robot_state.attached_collision_objects.push_back(attached_object);
    planning_scene.robot_state.is_diff = true;
    planning_scene_diff_publisher.publish(planning_scene);
  
    visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to continue the demo");
  
    // Detach an object from the robot
    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    // Detaching an object from the robot requires two operations
    //  * Detaching the object from the robot
    //  * Re-introducing the object into the environment
  
    /* First, define the DETACH object message*/
    moveit_msgs::AttachedCollisionObject detach_object;
    detach_object.object.id = "box";
    detach_object.link_name = "panda_hand";
    detach_object.object.operation = attached_object.object.REMOVE;
  
    // Note how we make sure that the diff message contains no other
    // attached objects or collisions objects by clearing those fields
    // first.
    /* Carry out the DETACH + ADD operation */
    ROS_INFO("Detaching the object from the robot and returning it to the world.");
    planning_scene.robot_state.attached_collision_objects.clear();
    planning_scene.robot_state.attached_collision_objects.push_back(detach_object);
    planning_scene.robot_state.is_diff = true;
    planning_scene.world.collision_objects.clear();
    planning_scene.world.collision_objects.push_back(attached_object.object);
    planning_scene.is_diff = true;
    planning_scene_diff_publisher.publish(planning_scene);
  
    visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to continue the demo");
  
    // Remove the object from the collision world
    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    // Removing the object from the collision world just requires
    // using the remove object message defined earlier.
    // Note, also how we make sure that the diff message contains no other
    // attached objects or collisions objects by clearing those fields
    // first.
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

## 6.2 Motion Planning Pipeline

