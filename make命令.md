# make

- 功能：读取Makefile,按里面所写的内容，将文件进行编译。
- make命令本身不知道如何做出该文件，需要知道一系列规则
- 编译compile：代码从高级语言 -> 汇编语言。 构建：编译的顺序安排(即上述的规则)
- 这一系列规则写在Makefile文件里。

# Makefile文件

- Makefile文件由一系列描述整个工程的编译和链接的规则所构成。   
  规则形式:
  ```
  <target> : <prerequisites>
  [tab] <commands>	
  ```
  - target:目标
    - 必须有，不可省略
    - 可以是文件名，也可以是某一个操作的名字(称之为伪目标)
  - prerequisites:前置条件
    - 非必须,但和command命令之间必须至少存在一个。
  - tab:空格
    - 必须在第二行前输入一个tab
  - commands:命令
    - 非必须,但和prerequisites前置条件之间必须至少存在一个
  - 例子：
  ```
  test:test.c
  	gcc -o test test.c
 
  #test为目标文件，也是我们最终生成的可执行文件。
  #test.c是依赖文件
  #gcc -o test test.c 重建目标文件 
  ```
- 编译过程中各文件的作用
	- .o文件（中间文件）
		- 作用：检查某个源文件是不是进行过修改，最终目标文件是不是需要重建
		- 由于会产生各种中间文件让文件夹看得很乱，所以要在Makefile文件的末尾清除中间文件
			```
			.PHONY:clean	#表明clean是一伪目标
			clean:
				rm -rf *.o test	#"*.o"是所有的中间文件，test是最终生成的执行文件
			```
- Makefile中的变量。
	- 定义和引用
	```
	VALUE_LIST = one two three
	
	OBJ=main.o test.o test1.o test2.o	#变量名OBJ
	test:$(OBJ)				#引用变量，$(OBJ)
      gcc -o test $(OBJ)
	```
	- 变量的赋值
		- :=,简单赋值。只对当前语句的变量有效
		- =，递归赋值。所有目标变量相关的其他变量都受影响
		- ?=条件赋值。如果未来未定义，则使用符号中的值定义变量
		- +=追加赋值。原变量用空格隔开的方式追加一个新值。
	
	


