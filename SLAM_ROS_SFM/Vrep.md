# Vrep 函数用法
## LUA 语言

1. 简言

   LUA语言是软件的自带语言，支持直接在软件进行编程
   
   -对某个物体添加脚本后便生成如下格式代码

   ```lua
   1.初始化函数，只在开始仿真时执行一次
   function sysCall_init()
    -- do some initialization here
   end

   2.执行函数，在仿真中循环执行
   function sysCall_actuation()
    -- put your actuation code here
   end

   3.传感器控制函数，--以后用到来补充一下，这个一般不用
   function sysCall_sensing()
    -- put your sensing code here
   end

   4.清除函数，停止运行前执行一次
   function sysCall_cleanup()
    -- do some clean-up here
   simUI.destroy(ui)	一般用来清除窗口
   end
   ```

2. 基本语法

   1. Lua的变量只要定义了并没有其他声明就是全局变量，比如你在一个函数中定义了，另一个函数也是共享的。

   2. 调用函数开头小写是lua语言调用，大写是其他语言接口

```lua
-- eg:
sim.getObjectAssociatedWithScript (Lua)                
simGetObjectAssociatedWithScript (C/C++)
```

3. 函数

   函数格式
```lua
function function_name()  
-- <函数体>   
end
```
括号中可以放一些传入的参数，需要返回值的时候直接return就好，跟C差不多，返回多个值的时候用逗号隔开

4. For循环
```lua
-- exp1为初值，exp2为终值，exp3为步长，可以省略，默认为1。
for var=exp1,exp2,exp3 
do  <执行体>  
end    
```

5. While循环
```lua
 -- condition为条件，比如x>1,
while （condition）
do <执行体> 
end 
```

6. If语句

6.1. if语句格式1
```lua
if （condition） 
 then <执行体> 
end 
```
6.2. if语句格式2
```lua
if （condition） 
 then  <执行体> 
 else <执行体> 
end  
```

7. 常用格式

数字类型
1. number 类型
大多数都是用这个好像

2. 数组
定义：num = {参数1，参数2…}
调用：num[1],num[2]即可 （注意索引从1开始，不是0）

3. 注释
Lua的注释使用的是两个减号–，这个可以注释一行代码，
批量注释使用的是–[[ 被注释内容 ]]-- 被包含的代码均会被注释

4. 打印输出
显示某个变量或是某段话，可以用print函数，

```lua
print(参数1，参数2...)
想要输出多个内容时需要用逗号隔开
```

5.创建UI
```lua
xml = '<ui title="'..sim.getObjectName(bubbleRobBase)..' speed" closeable="false" resizeable="false" activate="false">'	
	..[[
          <hslider minimum="0" maximum="100" on-change="speedChange_callback" id="1"/>
          <label text="" style="* {margin-left: 300px;}"/>
        </ui>
        ]]
    ui=simUI.create(xml)

```

6. dummy

- 在 V-REP（Virtual Robot Experimentation Platform）中，"dummy" 是一种特殊的虚拟对象类型，用于在场景中作为虚拟引用点或坐标系的标记点。Dummies 可以用于定位、指示和操作其他对象，而本身并不具有实际的物理特性。

- Dummies 具有以下特点：

位置和姿态: 每个 Dummy 都有自己的位置和姿态，可以用来指定其他对象的位置或方向。

父子关系: Dummies 可以成为其他对象的父对象，从而将它们连接在一起。子对象可以继承父对象的位置和方向。

方便引用: Dummies 可以用作对象的引用点，用于控制、移动或定位其他对象。

无碰撞: Dummies 是无碰撞的，它们不会与其他对象产生碰撞。

- 使用 V-REP 的 Dummy 对象，你可以：

将 Dummy 放置在机器人的特定部位，以便将其他对象附加到该部位，例如工具、末端执行器等。

将 Dummy 用作机器人的坐标系，用于执行变换操作，例如设置机器人的末端执行器在特定位置和方向。创建复杂的对象组合，通过将多个 Dummy 连接在一起。

要使用 Dummy 对象，通常的步骤包括：

- 在 V-REP 场景中创建一个 Dummy 对象。

将 Dummy 放置在适当的位置，并设置其姿态。

将其他对象连接到 Dummy 上，或使用 Dummy 来执行位置和姿态变换操作。

### API
#### sysCall_init() 
这个初始化的函数里面我们要获取YouBot机器人的各个关节的Handle

#### sim.setJointTargetVelocity() 
实现对某个指定轮子的控制过程
sim.setJointTargetVelocity(wheel_joints_name, wheel_velocity)

#### sim.getObjectHandle()
函数原型：number objectHandle=sim.getObjectHandle(string/number objectName)
功能：获取场景中某一个Object的句柄，这个句柄就是一个数字，在脚本中代表这个Object；
参数：一般为Object在场景中的名字，类型为string；
返回值：Object的句柄，类型为number;
