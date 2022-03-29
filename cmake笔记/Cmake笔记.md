http://c.biancheng.net/view/7112.html

# g++

- g++是c++的一个编译器

- 最简单的例子：

  命令行输入：

  ```
  g++ helloSlam.cpp
  ```

  会生成一个a.out的可执行文件：

  ```
  # 命令行输入如下指令来运行
  ./a.out 
  ```

- 任意一个c++程序都可以用g++来编译，但任意一个项目都可能含有十几个类，各类之间还存在复杂的依赖关系。这些类一部分要编译为可执行文件，一部分编译为库文件。

  所以如果单用g++，整个编译过程会异常繁琐。

  由此产生了方便的makefile以及更方便的cmake

- - 

# Makefile文件

## make

- 功能：读取Makefile,按里面所写的内容，将文件进行编译。
- make命令本身不知道如何做出该文件，需要知道一系列规则
- 编译compile：代码从高级语言 -> 汇编语言。 构建：编译的顺序安排(即上述的规则)
- 这一系列规则写在Makefile文件里。
- 使用：
  - `make <>`编译
  - `make clean`删除.o文件

## Makefile概述
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
  ```makefile
  test:test.c
	gcc -o test test.c
   
  #test为目标文件，也是我们最终生成的可执行文件。
  #test.c是依赖文件
  #gcc -o test test.c 重建目标文件 
  ```
  
- 编译过程和其生成的文件
	- 第一步:编译 complie
		- 1.第一个阶段：编译
			- 读取源代码(.c,.cpp)程序，先进行伪指令(#开头的）的编译，然后对其进行词法和语法分析，将高级语言指令转化为汇编代码
			- 预编译：文件先从.c转为转为.i文件。 gcc -E
			- 编译：文件从.i转为.s文件。 gcc -S
			- 后缀有: .s, .asm
		- 2.第二个阶段:汇编
		    - 将汇编语言翻译成目标机器指令
		    - 将.s文件转化成.o文件。 gcc -c
		    - 后缀有：.obj, .o, .a, .ko
	- 第二部:链接link
		- 将有关的目标彼此相连(因为代码中会引用其他源文件的函数/方法)
		- 从而使.o文件转化成可执行文件。 gcc
		- 后缀有：.exe, .elf, .axf
		
	- 涉及到的各种文件：
		- .c/.h文件
			- .h文件是头文件，内涵函数声明、宏定义、结构体定义等内容
			- .c文件时程序文件，内涵函数实现,变量定义等内容
			- 无本质区别	
		- .o文件（目标文件/中间文件）
			- 作用：检查某个源文件是不是进行过修改，最终目标文件是不是需要重建
			- 由于会产生各种中间文件让文件夹看得很乱，所以要在Makefile文件的末尾清除中间文件
			```makefile
			.PHONY:clean	#表明clean是一伪目标
			clean:
				rm -rf *.o test	#"*.o"是所有的中间文件，test是最终生成的执行文件
			```
	
## Makefile中的变量。

- 定义和引用
```makefile
VALUE_LIST = one two three
	
OBJ=main.o test.o test1.o test2.o	#变量名OBJ
test:$(OBJ)							#引用变量，$(OBJ)
    gcc -o test $(OBJ)
```
- 变量的赋值
	- :=,简单赋值。只对当前语句的变量有效
	- =，递归赋值。所有目标变量相关的其他变量都受影响
	- ?=条件赋值。如果未来未定义，则使用符号中的值定义变量
	- +=追加赋值。原变量用空格隔开的方式追加一个新值。

- 自动化变量

  - 规则的命令是对所有这一类文件的描述。我们在 Makefile 中描述规则时，依赖文件和目标文件是变动的，显然在命令中不能出现具体的文件名称，否则模式规则将失去意义。
  - 

  | 自动化变量 | 说明 |
  | ---------- | -------------- |
  | @         | 表示规则的目标文件名。如果目标是一个文档文件（Linux 中，一般成 .a 文件为文档文件，也称为静态的库文件），那么它代表这个文档的文件名。在多目标模式规则中，它代表的是触发规则被执行的文件名。 |
  | %         | 当目标文件是一个静态库文件时，代表静态库的一个成员名。       |
  | <         | 规则的第一个依赖的文件名。如果是一个目标文件使用隐含的规则来重建，则它代表由隐含规则加入的第一个依赖文件。 |
  | ?         | 所有比目标文件更新的依赖文件列表，空格分隔。如果目标文件时静态库文件，代表的是库文件（.o 文件）。 |
  | ^         | 代表的是所有依赖文件列表，使用空格分隔。如果目标是静态库文件，它所代表的只能是所有的库成员（.o 文件）名。一个文件可重复的出现在目标的依赖中，变量“\^”只记录它的第一次引用的情况。就是说变量“\^”会去掉重复的依赖文件。 |
  | +         | 类似“^”，但是它保留了依赖文件中重复出现的文件。主要用在程序链接时库的交叉引用场合。 |
  | *         | 在模式规则和静态模式规则中，代表“茎”。“茎”是目标模式中“%”所代表的部分（当文件名中存在目录时，“茎”也包含目录部分）。 |

  - 例子

  ```makefile
  test:test.o test1.o test2.o
           gcc -o $@ $^		#"$@" 代表的是目标文件test,“$^”代表的是依赖的文件
  test.o:test.c test.h
           gcc -o $@ $<		#“$<”代表的是依赖文件中的第一个
  test1.o:test1.c test1.h
           gcc -o $@ $<
  test2.o:test2.c test2.h
           gcc -o $@ $<
  # 执行 make 的时候，make 会自动识别命令中的自动化变量，并自动实现自动化变量中的值的替换
  ```


​	  

## Makefile目标文件搜索

如果需要的文件是存在于不同的路径下，在编译的时候要去怎么办？

- 两种搜索方法

  - VPATH：VPATH 是变量，更具体的说是环境变量，Makefile 中的一种特殊变量，使用时需要指定文件的路径；

    - 用法

    ```makefile
    VPATH := src car 	#多个路径的时候用空格或冒号隔开
    VPATH := src:car	#搜索的顺序为书写的顺序,先src再car
    ```

    - 实例

    ```makefile
    VPATH=src car
    test:test.o
        gcc -o $@ $^
    ```

  - vapth: vpath 是关键字，按照模式搜索，也可以说成是选择搜索。搜索的时候不仅需要加上文件的路径，还需要加上相应限制的条件。

    - 三种用法

    ```makefile
    1) vpath PATTERN DIRECTORIES #PATTERN：可以理解为要寻找的条件，
    2) vpath PATTERN			 #DIRECTORIES：寻找的路径 
    3) vpath
    ```

    - 实例

    ```makefile
    #第一种用法：
    vpath test.c src car 	 #多个路径的时候用空格或冒号隔开
    vpath test.c src：car    #在src和car下搜索test.c文件
    vapth %.c src 			 #搜索路径下所有.c文件
    #第二种用法：
    vpath test.c			 #清除符合文件 test.c 的搜索目录。
    #第三种用法
    vpath					 #清除所有已被设置的文件搜索路径。	
    ```

    

## Makefile条件判断

| **关键字** | **功能**                                            |
| ---------- | --------------------------------------------------- |
| ifeq       | 判断参数是否不相等，相等为 true，不相等为 false。   |
| ifneq      | 判断参数是否不相等，不相等为 true，相等为 false。   |
| ifdef      | 判断是否有值，有值为 true，没有值为 false。         |
| ifndef     | 判断是否有值，没有值为 true，有值为 false。         |
| else       | 表示当条件不满足的时候执行的部分                    |
| endif      | 是判断语句结束标志，Makefile 中条件判断的结束都要有 |

- 实例

  - ifeq,ifneq

  ```makefile
  libs_for_gcc= -lgnu
  normal_libs=
  ifeq($(CC),gcc)
      libs=$(libs_for_gcc)
  else
      libs=$(normal_libs)
  endif
  foo:$(objects)
      $(CC) -o foo $(objects) $(libs)
  ```

  - ifdef,ifndef

  ```makefile
  bar =
  foo = $(bar)
  all:
  ifdef foo
      @echo yes
  else
      @echo  no
  endif
  ```

## Makefile伪目标

- 目的
  - 避免我们的 Makefile 中定义的只执行的命令的目标和工作目录下的实际文件出现名字冲突。
  - 在 make 的并行和递归执行的过程中，此情况下一般会存在一个变量，定义为所有需要 make 的子目录。

- 声明伪目标

```makefile
.PHONY:clean
```



# cmake

## 作用

- 可以用cmake命令生成一个makefile文件。然后用make命令根据这个makefile文件的内容来编译整个工程。

## 流程

### 1. 编写代码 

并不是所有的代码都会编译为可执行文件

- 可执行程序：

  只有带有main函数的文件才会生成可执行程序。

  如：helloSlam.cpp

  ```c++
  #include "libHelloSlam.h"
  
  int main(int argc, char **argv){
      printHello();
      return 0;
  }
  ```

- 库：

  供其他程序调用用的代码。

  如：libHelloSlam.cpp

  ```c++
  #include <iostream>
  using namespace std;
  
  void printHello(){
      cout << "Hello Slam" << endl;
  }
  ```

  编译后会成为.a或.so文件。

- 头文件：

  库文件.a/.so是编译好的二进制文件，为了使用这个库，我们需要提供头文件。

  对于库的使用者，只要拿到了头文件和库文件就可以调用这个库。

  如：libHelloSlamh

  ```c++
  #ifndef LIBHELLOSLAM_H_
  #define LIBHELLOSLAM_H_
  //上面的宏定义是为了防止重复引用这个头文件而引起的重定义错误
  
  void printHello()
  
  #endif
  ```

### 2. 创建CMakeLists.txt

```cmake
# 声明已要求的cmake最低版本
cmake_minimum_required(VERSION 2.8)

# 声明一个cmake工程
project(HelloSLAM)

# 添加一个静态库
# 后缀名为.a
# 每次调用都会生成一个副本
add_library(hello libHelloSlam.cpp)

# 添加一个共享库
# 后缀名为.so
# 只有一个副本，更省空间
add_library(hello_shared SHARED libHelloSlam.cpp)

# 添加一个可执行程序
# 语法：add_executable(程序名 源代码文件)
add_executable(helloSlam helloSlam.cpp)

# 令helloSlam程序可以使用hello_shared库中的代码
target_link_libraries(helloSlam hello_shared)
```

- 我们对项目的编译管理工作，从输入一堆g++命令，转为了管理若干个比较直观的CMakeLists.txt文件。
- 每增加一个可执行文件，只要多加一行add_executable()
- cmake会帮我们自动解决代码的依赖关系

### 3. 创建makefile并make

在终端输入

```
# cmake命令会自动生成一堆中间文件
# 创建一个build子文件夹用来放置这些中间文件
# 这样要push代码的话，直接把build文件夹删了就行。
mkdir build			
cd build 
# 进入buid文件夹后，要编译的内容在上面一个目录
cmake ..
make
```
