# 零、学习资料汇总 

1. 学校script:https://www.moodle.tum.de/course/view.php?id=86463
2. http://beaglebone.cameon.net/home/autostarting-services
3. https://makezine.com/projects/get-started-with-beaglebone/


# 一、基本的输入输出
## 0. 编译
1. 点击Eclipse的锤子按钮编译
2. 编译好的文件是bin/Debug/EGR_run.bin
3. 插上单片机
4. 在终端输入`BBBsend 01_Beispielprogramm/bin/Debug/EGR_run.bin `
## 1. GPIO
GPIO(General Purpose In and Output)-Pins通用输入输出指的是可以接通和读取单片机数字信号的引脚。通常为高电平是5V/3.3V，低电平时0V。
## 2. 亮一颗LED灯
为了使用一个GPIO引脚来让一颗LED灯亮需要干三件事：1.Pin-Muxing 2.set Pin as output 3. turn pin on
### 2.1 Pin-Muxing引脚复用
- 设置引脚为GPIO的通用步骤
  1. 查看单片机引脚功能选择寄存器。
  2. 设置寄存器位。不同位代表不同功能，这里我们要设置为GPIO
- 在本例中：
  1. 要设置的引脚为第一区域的2号引脚引脚名字为LCD_DATA6，查表可知它对应的的寄存器为control-module下的LCD_DATA6配置寄存器。  
    control-module是cortex-A8 memory map下的一个region name。
    - 先根据这个region name找到一个基本地址，比如control-module就是0x44E10000。
    - 然后再找LCD_DATA6寄存器的相对于control-module起始位的地址：8B8h,h指的是16进制。
    - 最后将2地址相加就是LCD_DATA6的绝对地址，见下面代码。
  2. 设置LCD_DATA6为GPIO模式，根据表格查到GPIO在0-7的第7位。
- 代码实现：  
  ```c
  // 包含宏HWREG的头文件，可以直接对寄存器进行地址访问
  #include <hw_types.h> 

  // 引脚复用
  HWREG(0x44e10000+0x8b8) &=~ ((1<<0)|(1<<1)|(1<<2)); //删除之前的引脚位设置
  HWREG(0x44e10000+0x8b8) |= 0x7;	//选择引脚为第7位：GPIO模式
  ```
### 2.2 设置引脚为输出模式
经过2.1后，第一区域的引脚2已经设置为gpio模式，AM3359有4个GPIO引脚(gpio0-)，查表可知LCD_DATA6对应的gpio是gpio2_12：gpio2的第12个引脚。每一个GPIO引脚都有好多个寄存器，其中GPIO_OE(Output-Enable)就是用来控制gpio输入输出模式的。  
现在设置这个gpio为输出模式：
```c
// 查Memory Map可知region name为GPIO2的起始地址为0x481ac000
// 查GPIO registor可知GPIO_OE的地址为0x134
// 查GPIO_OE可知 设为0时为输出，设为1时为输入
// 因为要设置gpio2的第12个引脚为0，所以设置GPIO_OE寄存器的第12位=0
HWREG(0x481ac000+0x134) &= ~(1<<12)
```
### 2.3 控制引脚开关
经过2.2后，第一区域的引脚2已经设置为gpio的输出模式。现在用寄存器GPIO_DATAOUT来设置引脚开关（输出0v/3.3v）
```c
// 查Memory Map可知region name为GPIO2的起始地址为0x481ac000
// 查GPIO registor可知GPIO_DATAOUT的地址为0x13c
HWREG(0x481ac000+0x13c) &= ~(1<<12)
```
