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
## 一个例子

### 被动动力学的基本模拟C代码

```c
#include "mujoco.h"
#include "stdio.h"

char error[1000];
mjModel* m;
mjData* d;

int main(void)
{
   // load model from file and check for errors
   m = mj_loadXML("hello.xml", NULL, error, 1000);
   if( !m )
   {
      printf("%s\n", error);
      return 1;
   }

   // make data corresponding to model
   d = mj_makeData(m);

   // run simulation for 10 seconds
   while( d->time<10 )
      mj_step(m, d);

   // free model and data
   mj_deleteData(d);
   mj_deleteModel(m);

   return 0;
}
```

### xml例子

```xml
<mujoco model="example">
    <compiler coordinate="global"/>
    <default>
        <geom rgba=".8 .6 .4 1"/>
    </default>
    <asset>
        <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2=".6 .8 1"
                 width="256" height="256"/>
    </asset>
    <worldbody>
        <light pos="0 1 1" dir="0 -1 -1" diffuse="1 1 1"/>
        <body>
            <geom type="capsule" fromto="0 0 1  0 0 0.6" size="0.06"/>
            <joint type="ball" pos="0 0 1"/>
            <body>
                <geom type="capsule" fromto="0 0 0.6  0.3 0 0.6" size="0.04"/>
                <joint type="hinge" pos="0 0 0.6" axis="0 1 0"/>
                <joint type="hinge" pos="0 0 0.6" axis="1 0 0"/>
                <body>
                    <geom type="ellipsoid" pos="0.4 0 0.6" size="0.1 0.08 0.02"/>
                    <site name="end1" pos="0.5 0 0.6" type="sphere" size="0.01"/>
                    <joint type="hinge" pos="0.3 0 0.6" axis="0 1 0"/>
                    <joint type="hinge" pos="0.3 0 0.6" axis="0 0 1"/>
                </body>
            </body>
        </body>
        <body>
            <geom type="cylinder" fromto="0.5 0 0.2  0.5 0 0" size="0.07"/>
            <site name="end2" pos="0.5 0 0.2" type="sphere" size="0.01"/>
            <joint type="free"/>
        </body>
    </worldbody>
    <tendon>
        <spatial limited="true" range="0 0.6" width="0.005">
            <site site="end1"/>
            <site site="end2"/>
        </spatial>
    </tendon>
</mujoco>
```

