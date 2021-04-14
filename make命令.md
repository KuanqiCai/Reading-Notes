# make

- 功能：读取Makefile,按里面所写的内容，将文件进行编译。
- make命令本身不知道如何做出该文件，需要知道一系列规则
- 编译compile：代码从高级语言 -> 汇编语言。 构建：编译的顺序安排(即上述的规则)
- 这一系列规则写在Makefile文件里。

# Makefile文件

- Makefile文件由一系列规则构成。规则形式:
  - target:目标
    - 必须有，不可省略
    - 可以是文件名，也可以是某一个操作的名字(称之为伪目标)
  - prerequisites:前置条件
    - 非必须,但和command命令之间必须至少存在一个。
  - tab:空格
    - 必须在第二行前输入一个tab
  - commands:命令
    - 非必须,但和prerequisites前置条件之间必须至少存在一个

  ```makefile
  <target> : <prerequisites>
  [tab] <commands>	
  ```
- 
