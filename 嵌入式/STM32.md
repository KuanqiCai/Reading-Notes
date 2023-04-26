# 零、配置和资料

## 1. 配置环境

1. 下载MDK-Arm:https://www.keil.com/download/product/

2. 下载STM32芯片包（从[官网](http://www.keil.com/dd2/pack/)下载）
   - 我买的是STM32F103，搜索这个名字下载

## 2. 资料汇总

1. [Release Notes](file:///D:/Software/Keil_v5/ARM/HLP/Release_Notes.htm)



## 3. DAP仿真器下载程序

### 3.1 硬件连接

- 开发板/仿真器 通过usb线连接电脑
- 开发板和仿真器通过排线连接

### 3.2 仿真器配置

1. 打开一个project后，keil中打开魔法棒

2. Debug选项卡：

   - 选择CMSIS-DAP Debugger

     因为模拟器Fire-Debugger用的CMSIS-DAP标准

   - 打开CMSIS-DAP Debugger旁边的settings，中的debug

     - CMSIS-DAP: 连接仿真器，自动识别仿真器
     - SWDIO:连接仿真器和开发板后，自动识别芯片
     - PORT：选SW,速度5MHz

   - 打开CMSIS-DAP Debugger旁边的settings，中的flash download

     - 选择erase sectors,擦除比full快
     - 选择reset and run: 下载完后程序自动运行，不用手动复位
     - programming algorithm：点一下芯片

3. Utilities选项卡：默认的仿真器

4. 对于老的代码，需要安装v5编译器：[参考](https://blog.csdn.net/qq_62689333/article/details/129772909?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-129772909-blog-124890624.235%5Ev32%5Epc_relevant_default_base&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-129772909-blog-124890624.235%5Ev32%5Epc_relevant_default_base&utm_relevant_index=2)

### 3.3 下载运行

1. Keil上点击build
2. Keil上点击load