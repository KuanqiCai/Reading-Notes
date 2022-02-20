# 基本概念
## 介绍
- MuJoCo可用于基于模型的计算：

  - control synthesis控制合成，state estimation状态估计，system identification系统识别，mechanism design机械设计，通过逆动力学进行data analysis数据分析和parallel sampling for machine learning applications。

- 模型与数据分离：

  - `mjModel`包含模型描述，预计将保持不变
  - `mjData`包含所有动态变量和中间结果。作为一个scratch pad暂存器，所有函数从中读取输入和写入输出。它有一个预先分配的内部管理的栈，用来在运行时模块在模型初始化后不需要调用内存分配函数。
  - `mjModel`由编译器构建,`mjData`在运行时创建。top-level API函数反映了这种基本分离：`void mj_step(const mjModel* m, mjData* d);`

- 模型实例：

  |            | 高水平          | 低级             |
  | ---------- | --------------- | ---------------- |
  | 文件File   | MJCF/URDF(XML)  | MJB(二进制)      |
  | 储存Memory | mjCModel(c++类) | mjModel(C结构体) |

  三种获得mjModel的方法：(第二种尚未实现)

  1. (text editor)→ MJCF/URDF 文件 → (MuJoCo 解析器parser → mjCModel → MuJoCo 编译器compiler) → mjModel
  2. (用户代码) → mjCModel → (MuJoCo 编译器) → mjModel
  3. MJB 文件 →（MuJoCo 加载器loader）→ mjModel
  
- MuJoCo模拟器包含3部分

  1. STL文件，即三维模型
  2. XML文件，用于定义运动学和动力学关系
  3. 模拟器构建py文件，使用mujoco-py将XML model创建成可交互的环境，供强化学习算法调用。

## URDF/MJCF

[参考网站](https://mp.weixin.qq.com/s?__biz=MzI3MjI1NjMxMA==&mid=2247483652&idx=1&sn=0ea45ecfa99f74b4fe4e83b1839a57d3&chksm=eb341d5ddc43944b2d35ed6de7b61bb9d66c1f16e06d97380aa62f3944c104c24dae22825479&token=1676375003&lang=zh_CN#rd)

Unifed Robot Description Format统一机器人描述格式

- URDF的mesh文件包含
  - 材质颜色等信息的dae文件
  - 用于碰撞的stl文件
  - Mujoco只支持stl文件，所有的dae文件需要转为stl文件。可以使用MeshLab等软件进行转换
  
- URDF获得
  - 大公司的机器人，可通过官网或第三方网站获得
  - 自己设计的机器人，可以借助sw2urdf插件导出
  
- 将URDF转为XML文件（即MJCF）

  在mujoco的bin目录下有一个compile程序。

  `./compile model.urdf model.xml` 将其转为xml

  `./compile model.urdf model.txt` 将其转为txt

  `./compile model.urdf model.mjb` 将其转为mjb

- URDF和MJCF的区别

  MJCF相比于URDF有如下要求。（第三点为本质区别）

  1. 根元素mujoco,属性model为模型名
  2. 元素asset定义所有mesh为子元素
  3. 机器人全部建立在worldbody元素中
  4. 一个body包含子元素inertial、joint等。

## XML结构

[参考网站](https://mujoco.readthedocs.io/en/latest/XMLreference.html)

XML主要由3部分组成

- `<asset>`用<mesh>tag导入STL文件
- `<worldbody>`用<body>tag定义了所有的模拟器组件，包括灯光、地板以及机器人
- `<acutator>`：**定义可以执行运动的关节**。定义的顺序需要按照运动学顺序来，比如多关节串联机器人以工具坐标附近的最后一个关节为joint0，依此类推。

```xml
<!--根元素mujoco,属性modelel为模型名-->
<mujoco model="example">
    <!-- set some defaults for units and lighting -->
    <compiler angle="radian" meshdir="meshes"/>

    <!-- 导入STL文件 -->
    <asset>
        <mesh file="base.STL" />
        <mesh file="link1.STL" />
        <mesh file="link2.STL" />
    </asset>

    <!-- 定义所有模拟器组件 -->
    <worldbody>
        <!-- 灯光 -->
        <light directional="true" pos="-0.5 0.5 3" dir="0 0 -1" />
        <!-- 添加地板，这样我们就不会凝视深渊 -->
        <geom name="floor" pos="0 0 0" size="1 1 1" type="plane" rgba="1 0.83 0.61 0.5"/>
        <!-- the ABR Control Mujoco interface expects a hand mocap -->
        <body name="hand" pos="0 0 0" mocap="true">
            <geom type="box" size=".01 .02 .03" rgba="0 .9 0 .5" contype="2"/>
        </body>

        <!-- 构建串联机器人 -->
        <!--name: 实体名，pos: 与上一个实体的偏移量(推荐在body上设置，后面的pos都可以写作"0,0,0"，便于调试) (注意是相对位置！) -->
        <body name="base" pos="0 0 0">
            <!-- 定义其几何特性。引用导入的shoulder.STL模型并命名，可以用euler来旋转STL以实现两个实体间的配合 -->
            <geom name="link0" type="mesh" mesh="base" pos="0 0 0"/>
            <!-- 定义实体的惯性。如果不写inertial其惯性会从geom中推断出来 -->
            <inertial pos="0 0 0" mass="0" diaginertia="0 0 0"/>

            <!-- nest筑巢 each child piece inside the parent body tags -->
            <body name="link1" pos="0 0 1">
                <!-- this joint connects link1 to the base -->
                <!-- name: 关节名，pos: 与上一个实体的偏移量 -->
                <joint name="joint0" axis="0 0 1" pos="0 0 0"/>

                <geom name="link1" type="mesh" mesh="link1" pos="0 0 0" euler="0 3.14 0"/>
                <inertial pos="0 0 0" mass="0.75" diaginertia="1 1 1"/>

                <body name="link2" pos="0 0 1">
                    <!-- this joint connects link2 to link1 -->
                    <joint name="joint1" axis="0 0 1" pos="0 0 0"/>

                    <geom name="link2" type="mesh" mesh="link2" pos="0 0 0" euler="0 3.14 0"/>
                    <inertial pos="0 0 0" mass="0.75" diaginertia="1 1 1"/>

                    <!-- the ABR Control Mujoco interface uses the EE body to -->
                    <!-- identify the end-effector point to control with OSC-->
                    <body name="EE" pos="0 0.2 0.2">
                        <inertial pos="0 0 0" mass="0" diaginertia="0 0 0" />
                    </body>
                </body>
            </body>
        </body>
    </worldbody>

    <!-- 定义关节上的执行器 -->
    <actuator>
        <motor name="joint0_motor" joint="joint0"/>
        <motor name="joint1_motor" joint="joint1"/>
    </actuator>

</mujoco>
```

## MuJoCo控制器

- [mjdata结构](https://mujoco.readthedocs.io/en/latest/APIreference.html?highlight=mjdata#mjdata)中各种量的定义。

  - 其中有个mjtNum类型（即浮点数）的值ctrl，是机器人运动方程里的tao。可以通过设置这个值来施加控制
  - 所有actuator的控制输入都被保存在这个mjData.ctrl中
  - 所有的力输出都保存在mjData.actuator_force中
  - 所有的激活状态都保存在mjData.act中

- 官方给出的控制向量

  `u = (mjData.ctrl, mjData.qfrc_applied, mjData.xfrc_applied)`

  - ctrl:控制器的输入，单位是N.m，即一个力矩
  - qfrc_applied:关节空间施加的外力
  - xfrc_applied:笛卡尔空间施加的外力

  我们主要需要改动ctrl。后面两个量在考虑扰动时加入。

- 在xml中定义关节的控制范围

  ```xml
  <actuator>
      <!--对应于每个关节的电机-->
      <!--ctrllimted 说明控制是有范围的，这个范围由ctrlrange定义,gear是传动比-->
      <!--joint告知是哪一个关节，name告知这个关节的电机名字是什么-->
      <motor ctrllimted="true" ctrlrange="-150.0 150.0" joint="shoulde_pan_joint" name="torq_j1" gear= "101"/>
  	<motor ctrllimted="true" ctrlrange="-28.0 28.0" joint="wirst_1_joint" name="torq_j2"/>
  </actuator>
  ```

  可以直接在代码中将这个范围读出来

  ```python
  from mujoco_py import load_model_from_path
  ## robot.xml即模型的MJCF文件
  model = load_model_from_path('robot.xml')
  print(model.actuator_ctrlrange)
  ```

- sim.step()来进行步进式仿真（执行一次，计算一次）

  有三种[step()](https://mujoco.readthedocs.io/en/latest/APIreference.html?highlight=mj_step#mj-step)

  ```python
  sim=muijoco_py.MjSim(model)
  sim.step()
  ```

- 结合上面例子：

  ```python
  import mujoco_py as mp
  model = mp.load_model_from_path('ur5.xml')
  sim = mp.MjSim(model)
  ## MjViewer基于openGL实现，记录仿真，可用于展示仿真结果
  viewer = mp.MjViewer(sim)
  
  for i in range(3000):
      ## 每次循环给出1个控制输入（这里设置的控制是瞎写的）
      sim.data.crtl[:6] = 1
      sim.step()
      ## render显示仿真过程
      viewer.render()
  ```

  

## 使用mujoco-py

[github](https://github.com/openai/mujoco-py),[文档](https://openai.github.io/mujoco-py/build/html/index.html)

小模版

```python
from mujoco_py import load_model_from_path, MjSim
class my_env():
    def __init__(self, env, args):
        # super(lab_env, self).__init__(env)
        # 导入xml文档
        self.model = load_model_from_path("your_XML_path.xml")
        # 调用MjSim构建一个basic simulation
        self.sim = MjSim(model=self.model)

    def get_state(self, *args):
        self.sim.get_state()
        # 如果定义了相机
        # self.sim.data.get_camera_xpos('[camera name]')

    def reset(self, *args):
        self.sim.reset()

    def step(self, *args):
        self.sim.step()
```

