# 零、配置和资料

## 1. 配置环境

1. 下载MDK-Arm:https://www.keil.com/download/product/

2. 下载STM32芯片包（从[官网](http://www.keil.com/dd2/pack/)下载）
   - 我买的是STM32F103，搜索这个名字下载

## 2. 资料汇总

1. [视频学习](https://www.bilibili.com/video/BV1ct411S7QE/?spm_id_from=333.337.search-card.all.click&vd_source=4408d4cdc14f2a63048e7cc803b2da99)



## 3. DAP仿真器下载程序

为什么需要仿真器：

1. 下载程序（但也可以直接连usb线串口下载，所以是非主要功能）
2. 可以用于仿真，调试debug

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

##  4. 串口下载

- 下载安装CH340驱动（资料盘里有,win11可装）

  电脑USB <-> 单片机TTL

  - 他们不能直接通信，需要用到USB转串口模块：CH340
  - CH340用RXD和单片机的PA9引脚，TXD和单片机的PA10相连
  - PA9和PA10引脚即串口1，因为只有USART1才具有串口下载的功能

- 下载步骤：

  1. 打开mcuisp，串口下载软件
  2. 搜索串口：带ch340的
  3. 选择bps：115200
  4. 勾选校验 和 编程后执行，其他都不选
  5. 选择DTR的低电平复位，RTS高电平进BootLoaders
  6. 选择要上传的project的output文件夹/xx.hex
  7. 开始编程（注意要把仿真器拔了）

- 使用的ISP烧入程序：

  - ISP(In-System Programming)在系统可编程，不需要从电路板上取下芯片，也可以在已经编程的芯片上用ISP方式擦除和再烧入。
  - ISP通过芯片内部的自举程序（BootLoader，厂家烧好，无法更改）来选定一种串行的外设，对芯片内部FLASH进行编程

- 为什么ISP可以实现一键下载见[视频](https://www.bilibili.com/video/BV1ct411S7QE?p=3&spm_id_from=pageDriver&vd_source=4408d4cdc14f2a63048e7cc803b2da99)

## 5. STM32基础

### 5.1 STM32名字含义

- STM32字面含义：ST公司生产的32位微控制器
  - ST：意法半导体的公司名，是一家SOC厂商（做芯片的:买IP核+GPIO+RAM+FLASH），区别于ARM是IP厂商（生产IP内核的）
  - M：Microelectronics的缩写，表示微控制器
    - 区别于微处理器可以跑Linux，微控制器不能跑linux
  - 32：32bit

### 5.2 如何选型

原则：花最少的钱，做最多的事。

- 按如下顺序选择合适的MCU

  1. 选择内核：内核频率越高意味的功耗越高
  2. 选择引脚个数：引脚多少决定了资源的多少，也影响价格
  3. 选择多少RAM和多大FLASH：越多越贵
  4. 考虑该型号供货是否稳定

### 5.3 STM32分类

  ![img](https://raw.githubusercontent.com/Fernweh-yang/ImageHosting/main/img/202305160330544.png)

### 5.4 STM32命名规则

![img](https://raw.githubusercontent.com/Fernweh-yang/ImageHosting/main/img/202305160334079.png)

野火F103-MINI用的是STM32F103RCT6说明：

- F103: 是基础形
- R：说明引脚数右64个
- C：说明FLASH大小位256K
- T：说明封装方式为QFP
- 6：说明温度等级为A:-40~85度

选型时可以查看STM32官方的选型手册(资料盘里有)

### 5.5 如何寻找引脚/寄存器的功能说明

- 引脚查看官方资料：STM32FXXX数据手册，datasheet
- 寄存器查看官方资料：STM32FXXX参考手册，reference mannual