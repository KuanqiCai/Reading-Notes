# 一、Linux操作系统综述

## 1.一些比喻

- 软件外包公司-**操作系统**：

  操作系统其实就像一个软件外包公司，其内核就相当于这家外包公司的老板。

- 客户对接员-**输入设备驱动**：

  当客户（即输入设备）告诉对接员需求的时候，对于操作系统来讲，输入设备会发送一个中断。客户肯定希望外包公司停下手头的事来服务它。这个时候客户发送的需求就被称之为**中断事件interrupt event.**

- 交付人员-**输出设备驱动**

  作为外包公司，对客户的需求无论做或不做，做的结果如何，都要给客户一个反馈。例：显卡的显卡驱动。

- 立项-**运行软件**

  - 项目执行计划书：即已经编译的二进制执行程序，确定能做什么、怎么做、做的顺序是什么。
  - 档案管理系统-**文件管理子系统File Management Subsystem**：管理储存各个程序（他们都以二进制文件的形式保存在硬盘上）。二进制文件是静态的，称为program程序。运行起来的是动态的，称为process进程。
  - 项目管理系统-**进程管理子系统Process Management Subsystem**：进程的执行需要分配cpu来执行，即按照二进制代码一行一行的执行。该系统用于管理进程
  - 会议室管理系统-**内存管理子系统Memory Management Subsystem**：统一管理和分配不同进程所用到的内存空间。不同进程不共享内存，就像不同项目组不同时共用一个会议室

- 办事大厅-**系统调用System Call子系统**

  明文列出提供哪些服务(接口)，谁需要可以来申请，然后就会有回应。立项是办事大厅提供的关键服务之一。任何一个程序要想运行起来，就需要调用系统调用，创建进程。

- 对外合作部-**网络子系统**

  QQ 进程是不能直接发送网络包的，需要调用系统调用，内核使用网卡驱动程序进行发送。就像依靠公司的对外合作部和其他公司进行沟通合作。

### 内核源代码

https://elixir.bootlin.com/linux/latest/source

| 文件夹名字    | 内容                                                         |
| ------------- | ------------------------------------------------------------ |
| arch          | 架构相关，存放了许多cpu架构，比如arm,x86,MIPS,PPC            |
| block         | block表示块设备，以块(多个字节组成的整体)为单位来整体访问。比如sd卡、硬盘等都是块设备，可认为块设备就是存储设备。block目录下村粗着linux存储体系中关于块设备管理的代码。 |
| certs         | 与证书相关                                                   |
| crypto        | 加密。存放了一些常见的加密算法的C语言实现。比如crc32,md5,sha1。 |
| Documentation | 说明文档，对每个目录的具体作用进行说明                       |
| drivers       | 内核中所有设备的驱动程序，其中的每一个子目录对应一种设备驱动 |
| fs            | fs就是file system，文件系统，里面列出了linux支持的各种文件系统的实现。 |
| include       | 头文件目录，公共的（各种CPU架构共用的）头文件都在这里。每种CPU架构特有的一些头文件在arch/arm/include目录及其子目录下。 |
| init          | init是初始化的意思，这个目录下的代码就是linux内核启动时初始化内核的代码。 |
| ipc           | inter process commuication，进程间通信，里面都是linux支持的IPC的代码实现。 |
| kernel        | linux内核，所以这个文件夹下放的就是内核本身需要的一些代码文件。与处理器架构相关的内核代码在/kernel/$ARCH/kernel。 |
| lib           | 库。内核共用的函数库，与处理器架构相关的库在 /kernel/$ARCH/lib |
| mm            | 内存管理代码，譬如页式存储管理内存的分配和释放等。与具体处理器架构相关的内存管理代码位于/arch/$ARCH/mm目录下。 |
| net           | 网络相关的代码，譬如TCP/IP协议栈等都在这里。                 |
| samples       | 示例代码。                                                   |
| scripts       | 脚本，这个目录下全部是脚本文件.不是linux内核工作时使用的，而是用来辅助对linux内核进行配置编译生产的。用于实现内核配置的图形界面 |
| sound         | 与音频有关的代码，包括与音频有关的驱动程序                   |
| security      | 安全性相关的代码                                             |
| tools         | Linux中的常用工具                                            |
| usr           | 为内核尚未完全启动时执行用户空间代码提供了支持               |
| virt          | 此文件夹包含了虚拟化代码，它允许用户一次运行多个操作系统     |
| Kbuild        | Kbuild是kernel build的意思，就是内核编译的意思。这个文件就是linux内核特有的内核编译体系需要用到的文件。 |
| Kconfig       | 配置哪些文件编译，那些文件不用编译                           |
| CREDITS       | 贡献者列表                                                   |



## 2.系统调用

在源码中，系统调用定义于：对于 64 位操作系统unistd_64.h。

- 立项服务-**创建进程**

  - 创建进程的系统调用是**fork**，分支。要创建一个新的进程，需要一个老的进程调用 fork 来实现，其中老的进程叫作父进程（Parent Process），新的进程叫作子进程（Child Process）。
  - 当父进程调用 fork 创建进程的时候，子进程将各个子系统为父进程创建的数据结构也全部拷贝了一份，甚至连程序代码也是拷贝过来的。子进程需要做不同的事，就需要先判断哪个进程是子进程：

  - 对于 fork 系统调用的返回值，如果当前进程是子进程，就返回 0；如果当前进程是父进程，就返回子进程的进程号。通过if-else判断当前是父进程时，还接着做原来应该做的事情；是子进程时，需要请求另一个系统调用**execve**来执行另一个程序
  - 系统调用**waitpid**。父进程可以调用它，将子进程的进程号作为参数传给它，这样父进程就知道子进程运行完了没有，成功与否。

- 内存管理

  在操作系统中，每个进程都有自己的内存，互相之间不干扰，有独立的进程内存空间。

  - 代码段（Code Segment）：对于进程的内存空间来讲，放程序代码的这部分
  - 数据段（Data Segment）：对于进程的内存空间来讲，放进程运行中产生数据的这部分
    - 其中局部变量的部分，在当前函数执行的时候起作用，当进入另一个函数时，这个变量就释放了；
    - 也有动态分配的，会较长时间保存，指明才销毁的，这部分称为堆（Heap）。
  - 2个在堆里分配内存的系统调用
    - **brk**:当分配的内存数量比较小的时候，使用 brk，会和原来的堆的数据连在一起，这就像多分配两三个工位，在原来的区域旁边搬两把椅子就行了。
    - **mmap**:当分配的内存数量比较大的时候，使用 mmap，会重新划分一块区域，也就是说，当办公空间需要太多的时候，索性来个一整块。

- 文件管理

  Linux的一个特点：一切皆文件。Linux会为每一个文件分配一个文件描述符File Descriptor.常用的系统调用：

  - **open**打开文件，**close**关闭文件
  - **creat**创建文件
  - **lseek**打开文件跳到文件的某个位置
  - **read**读文件的内容，**write**写文件的内容

- 异常/信号处理

  当项目组收到信号的时候，项目组需要决定如何处理这些异常情况。

  对于一些不严重的信号，可以忽略，该干啥干啥，但是像 SIGKILL（用于终止一个进程的信号）和 SIGSTOP（用于中止一个进程的信号）是不能忽略的，会执行对于该信号的默认动作。

  - **sigaction**为某一个信号提供处理函数。

- 进程间通信

  - 首先就是发个消息，不需要一段很长的数据，这种方式称为**消息队列（Message Queue）**。
    - **msgget**创建一个新的队列
    - **msgsnd**将消息发送到消息队列
    - **msgrcv**从队列中取消息
  - 当两个项目组需要交互的信息比较大的时候，可以使用**共享内存**的方式，也即两个项目组共享一个会议室（这样数据就不需要拷贝来拷贝去）。
    - **shmget**创建一个共享的内存块
    - **shmat**将共享内存映射到自己的内存空间
      - 避免同时访问一块数据：**Semaphore**机制

- 网络通信

  - **Socket**建立一个Socket.

# 二、系统初始化

## 1. X86架构

X86架构是微处理器执行的计算机语言指令集，指一个intel通用计算机系列的标准编号缩写，也标识一套通用的计算机指令集合。

- **计算机工作模式**

  - CPU(Central Processing Unit 中央处理器)三部分组成：

    - 运算单元：加法、位移等计算操作
    - 数据单元：CPU内部的缓存和寄存器组。空间小但速度飞快。临时储存数据和运算结果
    - 控制单元：获得指令，并执行指令。
      - 指令指针寄存器：存放下一条指令在内存中的地址。因为进程的内存虽然隔离但并不连续。
      - 指令分两部分，一部分是做什么操作，例如是加法还是位移；一部分是操作哪些数据。执行这条指令，就要把第一部分交给运算单元，第二部分交给数据单元。

  - 总线Bus: CPU 和内存来来回回传数据，靠的都是总线。

    - 地址总线（Address Bus):传输地址数据，也就是我想拿内存中哪个位置的数据
      - 地址总线的位数，决定了能访问的地址范围到底有多广。例如只有两位，那 CPU 就只能认 00，01，10，11 四个位置，超过四个位置，就区分不出来了。位数越多，能够访问的位置就越多，能管理的内存的范围也就越广。
    - 数据总线（Data Bus):传输真正的数据
      - 数据总线的位数，决定了一次能拿多少个数据进来。例如只有两位，那 CPU 一次只能从内存拿两位数。要想拿八位，就要拿四次。位数越多，一次拿的数据就越多，访问速度也就越快。
  
- **8086PC**

    IBM用英特尔的8088 CPU开始做PC，而英特尔的这一系列CPU又开端于8086，所以称为X86架构：
    
    ![](https://static001.geekbang.org/resource/image/2d/1c/2dc8237e996e699a0361a6b5ffd4871c.jpeg)
    
    - 8086处理器由8个16位通用寄存器，用于计算工程中暂存数据
    
        - AX,BX,CX,DX,SP,BP,SI,DI
        - 其中AX,BX,CX,DX可以分成2个8位寄存器来使用，如：AH,AL。H代表高位，L代表低位。
    
    - **IP**寄存器 即 指令指针寄存器（Instruction Pointer Register）
        - 指向代码段中下一条指令的位置
        - CPU会根据此来不断的将指令从内存中提取到CPU的指令队列中去
        
    - 4个16位段寄存器
        - CS,DS,SS,ES。用于切换进程，指向不同进程的地址空间。每个进程都有各自的代码段和数据段。
        - **CS**:就是代码段寄存器（Code Segment Register），通过它可以找到代码在内存中的位置
        - DS 是数据段的寄存器，通过它可以找到数据在内存中的位置
          - 对于一个段，有一个起始的地址，而段内的具体位置，我们称为**偏移量（Offset）**。例如 8 号会议室的第三排，8 号会议室就是起始地址，第三排就是偏移量。
          - 寄存器都是16位的，但8086 的地址总线地址是 20 位。怎么凑够这 20 位呢？方法就是“起始地址 *16+ 偏移量”，也就是把 CS 和 DS 中的值左移 4 位，变成 20 位的，加上 16 位的偏移量，这样就可以得到最终 20 位的数据地址。
          - 无论真正的内存多么大，对于只有 20 位地址总线的 8086 来讲，能够区分出的地址也就 2^20=1M
          - 因为偏移量只能是 16 位的，所以一个段最大的大小是 2^16=64k。
        
        - SS 是栈寄存器（Stack Register）。栈是程序运行中一个特殊的数据结构，数据的存取只能从一端进行，秉承后进先出的原则，push 就是入栈，pop 就是出栈。
          - 凡是与函数调用相关的操作，都与栈紧密相关。例如，A 调用 B，B 调用 C。当 A 调用 B 的时候，要执行 B 函数的逻辑，因而 A 运行的相关信息就会被 push 到栈里面。当 B 调用 C 的时候，同样，B 运行相关信息会被 push 到栈里面，然后才运行 C 函数的逻辑。当 C 运行完毕的时候，先 pop 出来的是 B，B 就接着调用 C 之后的指令运行下去。B 运行完了，再 pop 出来的就是 A，A 接着运行，直到结束。
    
- 32位处理器

    在 32 位处理器中，有 32 根地址总线，可以访问 2^32=4G 的内存。在开放架构X86的基础上进行扩展：

    ![](https://static001.geekbang.org/resource/image/e3/84/e3f4f64e6dfe5591b7d8ef346e8e8884.jpeg)
    - 通用寄存器：
        - 将 8 个 16 位的扩展到 8 个 32 位的，但是依然可以保留 16 位的和 8 位的使用方式。不用将高16位分成2个8位，因为没有人写程序用高位的。。。
    - 段寄存器（Segment Register）：
        - 根据当时的硬件，没有把 16 位当成一个段的起始地址而是弄了一个不上不下的 20 位的地址，每次都要左移四位。如果新的段寄存器都改成 32 位的，明明 4G 的内存全部都能访问到，还左移不左移四位呢？
        - 所以从新定义：段的起始地址放在内存的某个地方。这个地方是一个表格，表格中的一项一项是**段描述符（Segment Descriptor**）。这里面才是真正的段的起始地址。而段寄存器里面保存的是在这个表格中的哪一项，称为**选择子（Selector）**。
            - 这样，将一个从段寄存器直接拿到的段起始地址，就变成了先间接地从段寄存器找到表格中的一项，再从表格中的一项中拿到段起始地址。
        - 保护模式下为了更快的拿到段起始地址，会将内存中的段描述符拿到CPU内的高速缓存中。
            - 实模式（Real Pattern）：段的起始地址是放在寄存器里面的，所以速度就比在内存里面快很多。当系统刚刚启动的时候，CPU 是处于实模式的
            - 保护模式（Protected Pattern）： 端起始地址放到内存里面了，就慢了，怎么办呢？将内存中的段描述符拿到CPU内的高速缓存中，就又快了。

## 2. 从BIOS到boot loader

- **BIOS**（Basic Input and Output System，基本输入输出系统）

  - 存储于ROM（Read Only Memory，只读存储器），区别于内存RAM（Random Access Memory，随机存取存储器）。是一个初始化程序。

    - 在 x86 系统中，将 1M 空间最上面的 0xF0000 到 0xFFFFF 这 64K 映射给 ROM，也就是说，到这部分地址访问的时候，会访问 ROM。

      ![](https://static001.geekbang.org/resource/image/5f/fc/5f364ef5c9d1a3b1d9bb7153bd166bfc.jpeg)

    - 当电脑刚加电的时候，会做一些重置的工作，将 CS (代码段寄存器Code Segment Register)设置为 0xFFFF，将 IP(指令指针寄存器Instruction Pointer Register)设置为 0x0000，所以第一条指令就会指向 0xFFFF0，正是在 ROM 的范围内。在这里，有一个 JMP 命令会跳到 ROM 中做初始化工作的代码，于是，BIOS 开始进行初始化的工作。

  - BIOS做3件事：

    1. 首先要检查下系统的各个硬件是不是都好着。
    2. 并建立一个中断向量表和中断服务程序，提供基本输入(中断)输出(显存映射)服务。
    3. 加载MBR**主引导记录**（Master Boot Record）到内存(0x7c00)
       - MBR又叫主引导扇区是计算机开机后访问硬盘时所必须要读取的首个扇区

- boot loader引导进程，位于电脑或其他计算机应用上，是指引导操作系统启动的进程。

  - 操作系统安装在硬盘上，在 BIOS 的界面上，可以看到一个启动盘的选项。
    - 启动盘一般在第一个扇区，占 512 字节，而且以 0xAA55 结束。这是一个约定，当满足这个条件的时候，就说明这是一个启动盘，在 512 字节以内会启动相关的代码。
  - 在 Linux 里有一个工具 Grub2(Grand Unified Bootloader Version 2),搞系统启动的
    - `grub2-mkconfig -o /boot/grub2/grub.cfg`配置系统启动的选项
    - `grub2-install /dev/sda`可以将启动程序安装到相应的位置。
  - MBR: 启动盘第一个扇区(512B, 由 Grub2 写入 boot.img 镜像)
    - boot.img 加载 Grub2 的 core.img 镜像
    - core.img 包括 diskroot.img, lzma_decompress.img, kernel.img 以及其他模块
    - boot.img 先加载运行 diskroot.img, 再由 diskroot.img 加载 core.img 的其他内容
    -  diskroot.img 解压运行 lzma_compress.img, 由lzma_compress.img 切换到保护模式

- 从实模式切换到保护模式需要做3件事

  1. **启用分段**，就是在内存里面建立段描述符表，将寄存器里面的段寄存器变成段选择子，指向某个段描述符，这样就能实现不同进程的切换了。辅助进程管理

  2. 启动分页。能够管理的内存变大了，就需要将内存分成相等大小的块。辅助内存管理

  3. 打开 Gate A20，也就是第 21 根地址线的控制线。

     - 在实模式 8086 下面，一共就 20 个地址线，只可访问 1M 的地址空间。

     - 打开了门后，就可以访问更宽的地址了

- zma_compress.img 解压运行 grub 内核 kernel.img, kernel.img 做以下四件事:

  1. 解析 grub.conf 文件

  2. 选择操作系统
  3. 例如选择 linux16, 会先读取内核头部数据进行检查, 检查通过后加载完整系统内核
  4. 启动系统内核

## 3. 内核初始化

从实模式切换到保护模式后，有了更强的寻址能力，就要开始启动内核了

- 内核的启动从入口函数 start_kernel() 开始。

  在 init/main.c 文件中，start_kernel 相当于内核的 main 函数。里面是各种各样初始化函数 XXXX_init。

  ![](https://static001.geekbang.org/resource/image/cd/01/cdfc33db2fe1e07b6acf8faa3959cb01.jpeg)

1. 首先初始化项目管理部门

   `set_task_stack_end_magic(&init_task)`创建**0号进程**

   - 参数`init_task`定义是`struct task_struct init_task = INIT_TASK(init_task)`
   - 它是系统创建的第一个进程，称为 **0 号进程**。这是唯一一个没有通过 fork 或者 kernel_thread 产生的进程，是**进程列表**的第一个。
   - 进程列表（Process List），就是咱们前面说的项目管理工具，里面列着我们所有接的项目。

2. 接着初始化办事大厅

   `tarp_init()`设置了很多中断门（Interrupt Gate），用于处理各种中断。

   - `set_system_intr_gate(IA32_SYSCALL_VECTOR, entry_INT80_32)`系统调用的中断门。系统调用也是通过发送中断的方式进行的

3. 初始化会议室管理系统

   `mm_init()`初始化内存管理模块。

4. 初始化项目管理调度

   `sched_init()`初始化调度模块

5. 文件系统初始化

   - `vfs_caches_init()`初始化基于内存的文件系统 rootfs

     - 这个函数里面，会调用 mnt_init()->init_rootfs() 这里有一行代码：`register_filesystem(&rootfs_fs_type)`。rootfs_fs_type是在 VFS 虚拟文件系统里面注册的一种类型，`struct file_system_type rootfs_fs_type。`
     - VFS（Virtual File System），虚拟文件系统。
       - 文件系统是我们的项目资料库，为了兼容各种各样的文件系统，我们需要将文件的相关数据结构和操作抽象出来，形成一个抽象层对上提供统一的接口。这个抽象层就是vfs

6. 最后start_kernel做其他初始化

   `rest_init()`做了很多初始化的工作

   1. 初始化 1 号进程（用户态所有进程的祖先）

      - 用 `kernel_thread(kernel_init, NULL, CLONE_FS)` 创建第二个进程，这个是 1 号进程

        - 有了其他进程后，就要区分哪些是核心资源，哪些是核心人员才能访问的核心保密区

        - x86提供了分层的权限机制，把区域分成了4个ring，越往里权限越高

          ![](https://static001.geekbang.org/resource/image/2b/42/2b53b470673cde8f9d8e2573f7d07242.jpg)

          - **内核态（Kernel Mode）**

            - 能够访问关键资源的代码放在 Ring0

          - **用户态（User Mode）**

            - 普通的程序代码放在 Ring3

            - 如果用户态的代码想要访问核心资源，就要通过系统调用。用户态代码不用管后面发生了什么，系统调用背后是内核态。

            - 当用户态程序运行到一半访问一个核心资源时，当前运行会停止并调用系统调用。暂停时，用户态程序运行代码到哪一行，当前的栈在哪里，这些信息都存在寄存器里。

                 ![](https://static001.geekbang.org/resource/image/71/e6/71b04097edb2d47f01ab5585fd2ea4e6.jpeg)

            - 调用过程：用户态 - 系统调用 - 保存寄存器 - 内核态执行系统调用 - 恢复寄存器 - 返回用户态，然后接着运行。

   2. 从内核态到用户态

      - 在1号进程启动的时候，还处在内核态，现在要转到用户态去运行一个程序。

        `kernel_thread`的参数是一个函数`kernel_init`，即1号进程会运行这个函数。这个函数会调用`kernel_init_freeable()`。里面有一代码：

        ```c
        if (!ramdisk_execute_command)
            ramdisk_execute_command = "/init";
        ```

        ramdisk在第3步，进入用户态后启用。

        - 1号进程`run_init_process()`函数会调用`do_execve`

          - execve 是一个系统调用，它的作用是运行一个执行文件。加一个 do_ 的往往是内核系统调用的实现。
          - 它会尝试运行 ramdisk 的“/init”，或者普通文件系统上的“/sbin/init”“/etc/init”“/bin/init”“/bin/sh”。不同版本的 Linux 会选择不同的文件启动，但是只要有一个起来了就可以。
          -  从内核态执行系统调用开始：do_execve->do_execveat_common->exec_binprm->search_binary_handler
            - 运行一个程序，在这里会加载一个二进制文件，linux下通常为ELF（Executable and Linkable Format，可执行与可链接格式）

        - 最后调用`start_thread`

          ```c
          
          void
          start_thread(struct pt_regs *regs, unsigned long new_ip, unsigned long new_sp){
          	set_user_gs(regs, 0);
          	regs->fs  = 0;
          	regs->ds  = __USER_DS;
          	regs->es  = __USER_DS;
          	regs->ss  = __USER_DS;
          	regs->cs  = __USER_CS;
          	regs->ip  = new_ip;
          	regs->sp  = new_sp;
          	regs->flags  = X86_EFLAGS_IF;
          	force_iret();
          }
          EXPORT_SYMBOL_GPL(start_thread);
          ```

          - `struct pt_regs`即寄存器，系统调用时在内核中保存用户态运行上下文

          - 用户态的代码段CS设置为`_USER_CS`

          - 用户态的数据段DS设置为`_USER_DS`

          - 指针寄存器IP

          - 栈指针寄存器SP

          - force_iret()：用于从系统调用中返回，恢复寄存器。

      - 到达用户态后：ramdisk的作用

        - 一开始到用户态的是 ramdisk 的 init，后来会启动真正根文件系统上的 init，成为所有用户态进程的祖先。
      
        - ramdisk是解决init程序的存储问题，在内核启动过程中需要init文件，如果从文件系统直接获取那么我们必须有各种磁盘的驱动才能从磁盘之上的文件系统读取到我们需要的文件，这样内核就复杂化啦，而采用ramdisk就是弱化磁盘驱动依赖，采用内存保存，这样就能直接启动。
        - 一开始运行 ramdisk 上的 /init。等它运行完了就已经在用户态了。/init 这个程序会先根据存储系统的类型加载驱动，有了驱动就可以设置真正的根文件系统了。有了真正的根文件系统，ramdisk 上的 /init 会启动文件系统上的 init。
        - 接下来就是各种系统的初始化。启动系统的服务，启动控制台，用户就可以登录进来了

   3. 创建2号进程（内核态所有线程运行的祖先）
   
      用户态的所有进程都有大师兄1号进程了。内核态的进程也需要一个进程统一管理：2号进程。
   
      - `kernel_thread(kthreadd, NULL, CLONE_FS | CLONE_FILES)`创建2号进程
        - 函数`kthreadd`负责所有内核态的线程的调度和管理，是内核态所有线程运行的祖先。
        - 为什么创建进程的函数名字叫kernel_thread()线程呢？因为从内核态来看，无论是进程，还是线程，我们都可以统称为任务（Task），都使用相同的数据结构，平放在同一个链表中。

## 4. 系统调用

**glibc** Linux提供的中介，将系统调用封装成更友好的接口。

https://www.gnu.org/software/libc/started.html

### glibc对系统调用的封装：

- 以系统调用open打开一个文件为例。我们调用的是glibc里面的open函数

`int open(const char *pathname, int flags, mode_t mode)`

- glibc有一个脚本make-syscall.sh，可以根据上面的配置文件对于每一个封装好的系统调用，生成一个文件。这个文件里面定义了一些宏，如：

  `define SYSCALL_NAME open`

- glibc有一个文件syscall-template.S ,使用了上面这个宏，定义了这个系统调用的调用方式。

  ```c
  
  T_PSEUDO (SYSCALL_SYMBOL, SYSCALL_NAME, SYSCALL_NARGS)
      ret
  T_PSEUDO_END (SYSCALL_SYMBOL)
  
  #define T_PSEUDO(SYMBOL, NAME, N)    PSEUDO (SYMBOL, NAME, N)
  ```

  - 这里的 PSEUDO 也是一个宏，它的定义如下

    ```c
    
    #define PSEUDO(name, syscall_name, args)                      \
      .text;                                      \
      ENTRY (name)                                    \
        DO_CALL (syscall_name, args);                         \
        cmpl $-4095, %eax;                               \
        jae SYSCALL_ERROR_LABEL
    ```

  - 里面对于任何一个系统调用，会调用 DO_CALL。这也是一个宏，这个宏 32 位和 64 位的定义是不一样的。

### 32位系统调用过程

sysdeps/unix/sysv/linux/i386目录下的sysdep.h文件

https://elixir.bootlin.com/glibc/glibc-2.34.9000/source/sysdeps/unix/sysv/linux/i386/sysdep.h#L77:

```c

/* Linux takes system call arguments in registers:
  syscall number  %eax       call-clobbered
  arg 1    %ebx       call-saved
  arg 2    %ecx       call-clobbered
  arg 3    %edx       call-clobbered
  arg 4    %esi       call-saved
  arg 5    %edi       call-saved
  arg 6    %ebp       call-saved
......
*/
#define DO_CALL(syscall_name, args)                           \
    PUSHARGS_##args                               \
    DOARGS_##args                                 \
    movl $SYS_ify (syscall_name), %eax;                          \
    ENTER_KERNEL                                  \
    POPARGS_##args
```

- 将请求参数放在寄存器里面，根据系统调用的名称，得到系统调用号，放在寄存器 eax 里面，然后执行 ENTER_KERNEL。

  - ENTER_KERNEL的定义：

    `# define ENTER_KERNEL int $0x80`

    int 就是 interrupt，也就是“中断”的意思。int $0x80 就是触发一个软中断，通过它就可以陷入（trap）内核。

- 内核启动的时候，有一个 trap_init()，其中有代码:

  `set_system_intr_gate(IA32_SYSCALL_VECTOR, entry_INT80_32);`

  这是一个软中断的陷入门，当接收到一个系统调用的时候，entry_INT80_32就被调用了。

  ```c
  
  ENTRY(entry_INT80_32)
          ASM_CLAC
          pushl   %eax                    /* pt_regs->orig_ax */
          SAVE_ALL pt_regs_ax=$-ENOSYS    /* save rest */
          movl    %esp, %eax
          call    do_syscall_32_irqs_on
  .Lsyscall_32_done:
  ......
  .Lirq_return:
    INTERRUPT_RETURN
  ```

  - 通过 push 和 SAVE_ALL 将当前用户态的寄存器，保存在 pt_regs 结构里面。

  - 进入内核之前，保存所有的寄存器，然后调用 do_syscall_32_irqs_on。它的实现如下

    ```c
    
    static __always_inline void do_syscall_32_irqs_on(struct pt_regs *regs)
    {
      struct thread_info *ti = current_thread_info();
      unsigned int nr = (unsigned int)regs->orig_ax;
    ......
      if (likely(nr < IA32_NR_syscalls)) {
        regs->ax = ia32_sys_call_table[nr](
          (unsigned int)regs->bx, (unsigned int)regs->cx,
          (unsigned int)regs->dx, (unsigned int)regs->si,
          (unsigned int)regs->di, (unsigned int)regs->bp);
      }
      syscall_return_slowpath(regs);
    }
    ```

    - 将系统调用号从 eax 里面取出来，然后根据系统调用号，在系统调用表中找到相应的函数进行调用，并将寄存器中保存的参数取出来，作为函数参数。这些参数所对应的寄存器，和 Linux 的注释是一样的。

      根据宏定义，`#define ia32_sys_call_table sys_call_table`，系统调用就是放在这个表里面

- 当系统调用结束之后，在 `entry_INT80_32` 之后，紧接着调用的是 `INTERRUPT_RETURN`，我们能够找到它的定义，也就是 `iret`

  `#define INTERRUPT_RETURN                iret`

  iret 指令将原来用户态保存的现场恢复回来，包含代码段、指令指针寄存器等。这时候用户态进程恢复执行。

- 总结：

  ![](https://static001.geekbang.org/resource/image/56/06/566299fe7411161bae25b62e7fe20506.jpg)

### 64位系统调用过程

/sysdeps/unix/sysv/linux/x86_64/sysdep.h

https://elixir.bootlin.com/glibc/glibc-2.34.9000/source/sysdeps/unix/sysv/linux/x86_64/sysdep.h

```c

/* The Linux/x86-64 kernel expects the system call parameters in
   registers according to the following table:
    syscall number  rax
    arg 1    rdi
    arg 2    rsi
    arg 3    rdx
    arg 4    r10
    arg 5    r8
    arg 6    r9
......
*/
#define DO_CALL(syscall_name, args)                \
  lea SYS_ify (syscall_name), %rax;                \
  syscall
```

- 将系统调用名称转换为系统调用号。相比较于32位，系统调用号会放到寄存器 rax；且不再是int中断而是改用syscall指令，减少了一次查表过程，性能有所提高。

  - syscall指令使用一种特殊的寄存器，叫做**特殊模块寄存器MSR(Model Specific Registers)**。是CPU为了完成某些特殊控制功能而用的寄存器，包括系统调用。

- 系统初始化的时候，trap_init会初始化上面的中断模式，并调用cpu_init->syscall_init.有代码：`wrmsrl(MSR_LSTAR, (unsigned long)entry_SYSCALL_64);`

  - rdmsr 和 wrmsr 是用来读写特殊模块寄存器的。MSR_LSTAR 就是这样一个特殊的寄存器，当 syscall 指令调用的时候，会从这个寄存器里面拿出函数地址来调用，也就是调用 entry_SYSCALL_64。

  - 在 arch/x86/entry/entry_64.S 中定义了 entry_SYSCALL_64

    ```c
    
    ENTRY(entry_SYSCALL_64)
            /* Construct struct pt_regs on stack */
            pushq   $__USER_DS                      /* pt_regs->ss */
            pushq   PER_CPU_VAR(rsp_scratch)        /* pt_regs->sp */
            pushq   %r11                            /* pt_regs->flags */
            pushq   $__USER_CS                      /* pt_regs->cs */
            pushq   %rcx                            /* pt_regs->ip */
            pushq   %rax                            /* pt_regs->orig_ax */
            pushq   %rdi                            /* pt_regs->di */
            pushq   %rsi                            /* pt_regs->si */
            pushq   %rdx                            /* pt_regs->dx */
            pushq   %rcx                            /* pt_regs->cx */
            pushq   $-ENOSYS                        /* pt_regs->ax */
            pushq   %r8                             /* pt_regs->r8 */
            pushq   %r9                             /* pt_regs->r9 */
            pushq   %r10                            /* pt_regs->r10 */
            pushq   %r11                            /* pt_regs->r11 */
            sub     $(6*8), %rsp                    /* pt_regs->bp, bx, r12-15 not saved */
            movq    PER_CPU_VAR(current_task), %r11
            testl   $_TIF_WORK_SYSCALL_ENTRY|_TIF_ALLWORK_MASK, TASK_TI_flags(%r11)
            jnz     entry_SYSCALL64_slow_path
    ......
    entry_SYSCALL64_slow_path:
            /* IRQs are off. */
            SAVE_EXTRA_REGS
            movq    %rsp, %rdi
            call    do_syscall_64           /* returns with IRQs disabled */
    return_from_SYSCALL_64:
      RESTORE_EXTRA_REGS
      TRACE_IRQS_IRETQ
      movq  RCX(%rsp), %rcx
      movq  RIP(%rsp), %r11
        movq  R11(%rsp), %r11
    ......
    syscall_return_via_sysret:
      /* rcx and r11 are already restored (see code above) */
      RESTORE_C_REGS_EXCEPT_RCX_R11
      movq  RSP(%rsp), %rsp
      USERGS_SYSRET64
    ```

    - 先保存了很多寄存器到 pt_regs 结构里面，例如用户态的代码段、数据段、保存参数的寄存器，然后调用 entry_SYSCALL64_slow_pat->do_syscall_64:

      ```c
      
      __visible void do_syscall_64(struct pt_regs *regs)
      {
              struct thread_info *ti = current_thread_info();
              unsigned long nr = regs->orig_ax;
      ......
              if (likely((nr & __SYSCALL_MASK) < NR_syscalls)) {
                      regs->ax = sys_call_table[nr & __SYSCALL_MASK](
                              regs->di, regs->si, regs->dx,
                              regs->r10, regs->r8, regs->r9);
              }
              syscall_return_slowpath(regs);
      }
      ```

      - 在 do_syscall_64 里面，从 rax 里面拿出系统调用号，然后根据系统调用号，在系统调用表 sys_call_table 中找到相应的函数进行调用，并将寄存器中保存的参数取出来，作为函数参数。
      - **与32位一样，都会到系统调用表sys_call_table这来。**

- 64 位的系统调用返回的时候，执行的是 USERGS_SYSRET64。定义如下：

  ```c
  
  #define USERGS_SYSRET64        \
    swapgs;          \
    sysretq;
  ```

  返回用户态的指令变成了 sysretq。

- 总结：

  https://static001.geekbang.org/resource/image/1f/d7/1fc62ab8406c218de6e0b8c7e01fdbd7.jpg

### 系统调用表

- 32 位的系统调用表定义在 arch/x86/entry/syscalls/syscall_32.tbl 文件里,如open

  `5  i386  open      sys_open  compat_sys_open`

- 64 位的系统调用定义在另一个文件 arch/x86/entry/syscalls/syscall_64.tbl 里,如open

  `2  common  open      sys_open`

  - 第一列的数字是系统调用号。可以看出，32 位和 64 位的系统调用号是不一样的
  - 第三列是系统调用的名字
  - 第四列是系统调用在内核的实现函数。不过，它们都是以 sys_ 开头

- 系统调用在内核中的实现函数要有一个声明。声明往往在 include/linux/syscalls.h 文件中。例如 sys_open 是这样声明的：

  ```c
  
  asmlinkage long sys_open(const char __user *filename,
                                  int flags, umode_t mode);
  ```

  真正的实现这个系统调用，一般在一个.c 文件里面，例如 sys_open 的实现在 fs/open.c 里面:

  ```c
  
  SYSCALL_DEFINE3(open, const char __user *, filename, int, flags, umode_t, mode)
  {
          if (force_o_largefile())
                  flags |= O_LARGEFILE;
          return do_sys_open(AT_FDCWD, filename, flags, mode);
  }
  ```

  - SYSCALL_DEFINE3 是一个宏系统调用最多六个参数，根据参数的数目选择宏。具体是这样定义的：

    ```c
    
    #define SYSCALL_DEFINE1(name, ...) SYSCALL_DEFINEx(1, _##name, __VA_ARGS__)
    #define SYSCALL_DEFINE2(name, ...) SYSCALL_DEFINEx(2, _##name, __VA_ARGS__)
    #define SYSCALL_DEFINE3(name, ...) SYSCALL_DEFINEx(3, _##name, __VA_ARGS__)
    #define SYSCALL_DEFINE4(name, ...) SYSCALL_DEFINEx(4, _##name, __VA_ARGS__)
    #define SYSCALL_DEFINE5(name, ...) SYSCALL_DEFINEx(5, _##name, __VA_ARGS__)
    #define SYSCALL_DEFINE6(name, ...) SYSCALL_DEFINEx(6, _##name, __VA_ARGS__)
    
    
    #define SYSCALL_DEFINEx(x, sname, ...)                          \
            SYSCALL_METADATA(sname, x, __VA_ARGS__)                 \
            __SYSCALL_DEFINEx(x, sname, __VA_ARGS__)
    
    
    #define __PROTECT(...) asmlinkage_protect(__VA_ARGS__)
    #define __SYSCALL_DEFINEx(x, name, ...)                                 \
            asmlinkage long sys##name(__MAP(x,__SC_DECL,__VA_ARGS__))       \
                    __attribute__((alias(__stringify(SyS##name))));         \
            static inline long SYSC##name(__MAP(x,__SC_DECL,__VA_ARGS__));  \
            asmlinkage long SyS##name(__MAP(x,__SC_LONG,__VA_ARGS__));      \
            asmlinkage long SyS##name(__MAP(x,__SC_LONG,__VA_ARGS__))       \
            {                                                               \
                    long ret = SYSC##name(__MAP(x,__SC_CAST,__VA_ARGS__));  \
                    __MAP(x,__SC_TEST,__VA_ARGS__);                         \
                    __PROTECT(x, ret,__MAP(x,__SC_ARGS,__VA_ARGS__));       \
                    return ret;                                             \
            }                                                               \
            static inline long SYSC##name(__MAP(x,__SC_DECL,__VA_ARGS__)
    ```

  - 把宏展开之后，实现如下，和声明的是一样的。

    ```c
    
    asmlinkage long sys_open(const char __user * filename, int flags, int mode)
    {
     long ret;
    
    
     if (force_o_largefile())
      flags |= O_LARGEFILE;
    
    
     ret = do_sys_open(AT_FDCWD, filename, flags, mode);
     asmlinkage_protect(3, ret, filename, flags, mode);
     return ret;
    ```

- 声明和实现都好了。接下来，在编译的过程中，需要根据 syscall_32.tbl 和 syscall_64.tbl 生成自己的 unistd_32.h 和 unistd_64.h。生成方式在 arch/x86/entry/syscalls/Makefile 中。

  - 这里面会使用两个脚本:

  1. 第一个脚本 arch/x86/entry/syscalls/syscallhdr.sh，会在文件中生成 #define __NR_open
  2. 第二个脚本 arch/x86/entry/syscalls/syscalltbl.sh，会在文件中生成 \__SYSCALL(__NR_open, sys_open)。

- 在文件 arch/x86/entry/syscall_32.c，定义了这样一个表，里面 include 了这个头文件，从而所有的 sys_ 系统调用都在这个表里面了。

  ```c
  __visible const sys_call_ptr_t ia32_sys_call_table[__NR_syscall_compat_max+1] = {
          /*
           * Smells like a compiler bug -- it doesn't work
           * when the & below is removed.
           */
          [0 ... __NR_syscall_compat_max] = &sys_ni_syscall,
  #include <asm/syscalls_32.h>
  };
  ```

- 同理，在文件 arch/x86/entry/syscall_64.c，定义了这样一个表，里面 include 了这个头文件，这样所有的 sys_ 系统调用就都在这个表里面了。

  ```c
  /* System call table for x86-64. */
  asmlinkage const sys_call_ptr_t sys_call_table[__NR_syscall_max+1] = {
    /*
     * Smells like a compiler bug -- it doesn't work
     * when the & below is removed.
     */
    [0 ... __NR_syscall_max] = &sys_ni_syscall,
  #include <asm/syscalls_64.h>
  };
  ```

- 总结

  ![64位](https://static001.geekbang.org/resource/image/86/a5/868db3f559ad08659ddc74db07a9a0a5.jpg)

# 三、 进程管理

## 1.  进程

### 代码：用系统调用创建进程

- 一个创建进程的函数

  ```c
  #include <stdio.h>
  #include <stdlib.h>
  #include <sys/types.h>
  #include <unistd.h>
        
  extern int create_process (char* program, char** arg_list);
          
  int create_process (char* program, char** arg_list)
  {
      pid_t child_pid;
      child_pid = fork ();
      if (child_pid != 0)
          return child_pid;
      else {
          execvp (program, arg_list);
          abort ();
      }
  }
  ```

  - 这里的if-else根据fork返回值的不同，父进程和子进程就分道扬镳了。
  - 在子进程中，用execvp运行一个新的程序。

- 调用上面这个函数

  ```c
  #include <stdio.h>
  #include <stdlib.h>
  #include <sys/types.h>
  #include <unistd.h>
  
  extern int create_process (char* program, char** arg_list);
  
  int main ()
  {
      char* arg_list[] = {
          "ls",
          "-l",
          "/etc/yum.repos.d/",
          NULL
      };
      create_process ("ls", arg_list);
      return 0;
  }
  ```

  - 用子进程运行ls命令

### 编译compile

将程序编译位二进制文件，其有严格的格式ELF（Executeable and Linkable Format可执行与可链接格式）有**几种ELF的格式：**

![](https://static001.geekbang.org/resource/image/85/de/85320245cd80ce61e69c8391958240de.jpeg)

#### 1. .o文件：可重定位文件（Relocatable File)，

```c
gcc -c -fPIC process.c
gcc -c -fPIC createprocess.c
    
## -fPIC 作用于编译阶段，告诉编译器产生与位置无关代码(Position-Independent Code)，则产生的代码中，没有绝对地址，全部使用相对地址，故而代码可以被加载器加载到内存的任意位置，都可以正确的执行。这正是共享库所要求的，共享库被加载时，在内存的位置不是固定的。
```

![](https://static001.geekbang.org/resource/image/e9/d6/e9c2b4c67f8784a8eec7392628ce6cd6.jpg)

- 文件格式为：
  - ELF 文件的头是用于描述整个文件的
  - .text：放编译好的二进制可执行代码
  - .data：已经初始化好的全局变量
  - .rodata：只读数据，例如字符串常量、const 的变量
  - .bss：未初始化全局变量，运行时会置 0
  - .symtab：符号表，记录的则是函数和变量
  - .strtab：字符串表、字符串常量和变量名
  - .rel.text：标注某个函数是需要重定位的。比如当前函数a调用了另一个函数b，但b在另一个.o文件里，那么当前.o文件不知道被调用函数的位置，只能在这里先标注b函数，之后再定位。
  - 节头部表（Section Header Table）：保存前面各个节section的元数据信息。

- .a文件Archives**静态链接库**。让 create_process 这个函数作为库文件被重用

	```
	ar cr libstaticprocess.a process.o
  ```

	虽然这里 libstaticprocess.a 里面只有一个.o，但是实际情况可以有多个.o。当有程序要使用这个静态连接库的时候，会将.o 文件提取出来，链接到程序中。

	```
	gcc -o staticcreateprocess createprocess.o -L. -lstaticprocess
	```

	- -L 表示在当前目录下找.a 文件
	- -lstaticprocess 会自动补全文件名，比如加前缀 lib，后缀.a，变成 libstaticprocess.a
	- 找到这个.a 文件后，将里面的 process.o 取出来，和 createprocess.o 做一个链接，形成二进制执行文件 **staticcreateprocess**。
	- 这个链接的过程，重定位就起作用了，原来 createprocess.o 里面调用了 create_process 函数，但是不能确定位置，现在将 process.o 合并了进来，就知道位置了。

#### 2. **可执行文件**,即上面形成的2进制文件staticcreateprocess,ELF的第二种格式。

![](https://static001.geekbang.org/resource/image/1d/60/1d8de36a58a98a53352b40efa81e9660.jpg)

- 这里的section是多个.o文件合并过的，可以马上就加载到内存里去。代码段和数据段会被加载到内存里，其他的不会。

- 小的section被合并成了大的段segment，并再最前面加了一个段头表(Segment Header Table)。里面除了有对于段的描述外，还有p_vaddr是这个段加载到内存的虚拟地址。

  运行后：

  ```
  yang@yang:~/Desktop/test$ ./staticcreateprocess 
  yang@yang:~/Desktop/test$ total 2097256
  drwxr-xr-x   2 root root       4096 Sep 16 10:53 bin
  ....
  ```

  

- 静态链接库和动态链接库：

  - 静态链接库一旦链接进去，代码和变量的 section 都合并了，因而程序运行的时候，就不依赖于这个库是否存在。
    - 缺点，就是相同的代码段，如果被多个程序使用的话，在内存里面就有多份，而且一旦静态链接库更新了，如果二进制执行文件不重新编译，也不随着更新。
  - 动态链接库（Shared Libraries）多个对象文件的重新组合，可被多个程序共享。

  ```
  gcc -shared -fPIC -o libdynamicprocess.so process.o
  ```

  当一个动态链接库被链接到一个程序文件中的时候，最后的程序文件并不包括动态链接库中的代码，而仅仅包括对动态链接库的引用，并且不保存动态链接库的全路径，仅仅保存动态链接库的名称。

  ```
  gcc -o dynamiccreateprocess createprocess.o -L. -ldynamicprocess
  ```

  当运行这个程序的时候，首先寻找动态链接库，然后加载它。默认情况下，系统在 /lib 和 /usr/lib 文件夹下寻找动态链接库。如果找不到就会报错，我们可以设定 LD_LIBRARY_PATH 环境变量，程序运行时会在此环境变量指定的文件夹下寻找动态链接库。

  运行后：

  ```
  yang@yang:~/Desktop/test$ export LD_LIBRARY_PATH=.
  yang@yang:~/Desktop/test$ ./dynamiccreateprocess 
  yang@yang:~/Desktop/test$ total 2097256
  drwxr-xr-x   2 root root       4096 Sep 16 10:53 bin
  ```

#### 3.动态链接库，即ELF的第三种类型：**共享对象文件**Shared Object

- 相比静态连裤生成的二进制文件格式ELF，多了一个.interp的段Segment，这里面是ld-linux.so动态链接器，也就是说，运行时的链接动作都是它做的。
- 还多了2个section,一个是.plt，过程链接表（Procedure Linkage Table，**PLT**），一个是.got.plt，全局偏移量表（Global Offset Table，**GOT**）。

  - dynamiccreateprocess 这个程序要调用 libdynamicprocess.so 里的 create_process 函数。由于是运行时才去找，编译的时候，压根不知道这个函数在哪里，所以就在 **PLT** 里面建立一项 PLT[x]。

    - 这一项也是一些代码，有点像一个本地的代理，在二进制程序里面，不直接调用 create_process 函数，而是调用 PLT[x]里面的代理代码，这个代理代码会在运行的时候找真正的 create_process 函数。

  - 去哪里找代理代码呢？这就用到了 **GOT**，这里面也会为 create_process 函数创建一项 GOT[y]。这一项是运行时 create_process 函数在内存中真正的地址。

    - 程序调用共享库里的create_process的时候，调用的是对应代理PLT[x]，PLT[x]再去调用GOT[y]。GOT[y]对应的就是create_process在内存中的位置

    - GOT[y]的值哪来的呢？：对于 create_process 函数，GOT 一开始就会创建一项 GOT[y]，但是这里面没有真正的地址需要回调PLT来找create_process 函数的真实地址.

      PLT 这个时候会转而调用 PLT[0]，也即第一项，PLT[0]转而调用 GOT[2]，这里面是 ld-linux.so 的入口函数，这个函数会找到加载到内存中的libdynamicprocess.so 里面的 create_process 函数的地址，然后把这个地址放在 GOT[y]里面。下次，PLT[x]的代理函数就能够直接调用了。

  - 为什么绕来绕去？因为要统一PLT和GOT的作用，PLT就是用来放代理代码的，也即stub代码的，GOT是用来存放so对应的真实代码的地址的。

    ld-linux.so虽然默认会被加载，但是也是一个so，所以会放在GOT里面。要调用这个so里面的代码，也是需要从stub里面统一调用进去的，所以要回到PLT去调用。

### 查看各个进程

linux内核有如下数据结构来定义加载二进制文件到内存中的方法：

```c
struct linux_binfmt {
        struct list_head lh;
        struct module *module;
        int (*load_binary)(struct linux_binprm *);
        int (*load_shlib)(struct file *);
        int (*core_dump)(struct coredump_params *cprm);
        unsigned long min_coredump;     /* minimal dump size */
} __randomize_layout;
```

对ELF文件格式：

```c
static struct linux_binfmt elf_format = {
        .module         = THIS_MODULE,
        .load_binary    = load_elf_binary,
        .load_shlib     = load_elf_library,
        .core_dump      = elf_core_dump,
        .min_coredump   = ELF_EXEC_PAGESIZE,
};
```

用`ps -ef`命令查看当前系统启动的进程，有三类

```shell
yang@yang:~/Desktop/test$ ps -ef
UID         PID   PPID  C STIME TTY          TIME CMD
root          1      0  0 11:01 ?        00:00:01 /sbin/init splash
root          2      0  0 11:01 ?        00:00:00 [kthreadd]
yang       1705      1  0 11:01 ?        00:00:00 /lib/systemd/systemd --user
yang       3593   1705  0 13:28 ?        00:00:01 /usr/lib/gnome-terminal/gnome-
yang       3602   3593  0 13:28 pts/0    00:00:00 bash
yang       3873   3602  0 14:12 pts/0    00:00:00 ps -ef

```

- PID 1 的进程就是我们的 init 进程 init splash。用户态不带中括号
- PID 2 的进程是内核线程 kthreadd。内核态带中括号
- tty那一列是问号的说明不是前台启动的，一般都是后台的服务
- PPID告诉了这个进程是从哪一个进程fork而来的。

## 2.线程

### 为什么要有线程

- 任何一个进程默认有一个主线程，线程负责执行二进制命令。进程除了执行指令外，还要管理内存和文件系统等。所以进程相当于一个项目，而线程就是为了完成项目需求，而建立的一个个开发任务。
- 对于并行任务，进程有2个问题：第一，创建进程占用资源太多；第二，进程之间的通信需要数据在不同的内存空间传来传去，无法共享。
- 多线程还能留出1个单独的线程处理突发事件，不用让主线程停下来去处理。

### 创建线程

```c

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_OF_TASKS 5

void *downloadfile(void *filename)
{
   printf("I am downloading the file %s!\n", (char *)filename);
   sleep(10);
   long downloadtime = rand()%100;
   printf("I finish downloading the file within %d minutes!\n", downloadtime);
   pthread_exit((void *)downloadtime); //pthread_exit退出线程，传入一个参数转为void*类型作为线程退出的返回值
}

int main(int argc, char *argv[])
{
   char files[NUM_OF_TASKS][20]={"file1.avi","file2.rmvb","file3.mp4","file4.wmv","file5.flv"};
   pthread_t threads[NUM_OF_TASKS]; //一个有5个pthread_t线程对象的数组
   int rc;
   int t;
   int downloadtime;

   pthread_attr_t thread_attr;	//线程属性
   pthread_attr_init(&thread_attr);	//初始化这个线程属性
   pthread_attr_setdetachstate(&thread_attr,PTHREAD_CREATE_JOINABLE);//设置属性为PTHREAD_CREATE_JOINABLE，表示将来主线程程等待这个线程的结束，并获取退出时的状态
	
   /*对每一个任务用pthread_create创建一个线程*/
   for(t=0;t<NUM_OF_TASKS;t++){
     printf("creating thread %d, please help me to download %s\n", t, files[t]);
     rc = pthread_create(&threads[t], &thread_attr, downloadfile, (void *)files[t]);//1.线程对象2.线程属性3.线程运行函数4，线程运行函数的参数
     if (rc){
       printf("ERROR; return code from pthread_create() is %d\n", rc);
       exit(-1);
     }
   }

   pthread_attr_destroy(&thread_attr);//销毁线程属性

   for(t=0;t<NUM_OF_TASKS;t++){
     pthread_join(threads[t],(void**)&downloadtime);	//停止线程，主线程接收与pthread_t类型threads所关联的返回值downloadtime
     printf("Thread %d downloads the file %s in %d minutes.\n",t,files[t],downloadtime);
   }

   pthread_exit(NULL);
}
```

编译：

`gcc download.c -lpthread`

### 线程的数据

- 线程访问的数据有3类：

  1. 线程栈上的本地数据。比如函数执行过程中的局部变量

     - 栈的大小可以通过命令 ulimit -a 查看，默认情况下线程栈大小为 8192（8MB）。我们可以使用命令 ulimit -s 修改。

     - 修改线程栈的大小：

       `int pthread_attr_setstacksize(pthread_attr_t *attr, size_t stacksize);`

     - 主线程在内存中有一个栈空间，其他线程栈也拥有独立的栈空间。为了避免线程之间的栈空间踩踏，线程栈之间还会有小块区域，用来隔离保护各自的栈空间。

  2. 在整个进程里共享的全局数据。比如全局变量

  3. 线程私有数据Thread Specific Data

     - 创建一个key,伴随一个析构函数。等到线程退出的时候，就会调用析构函数释放 value。

       `int pthread_key_create(pthread_key_t *key, void (*destructor)(void*))`

       key 一旦被创建，所有线程都可以访问它，但各线程可根据自己的需要往 key 中填入不同的值，这就相当于提供了一个同名而不同值的全局变量。

     - 设置 key 对应的 value。

       `int pthread_setspecific(pthread_key_t key, const void *value)`

     - 获取 key 对应的 value

       `void *pthread_getspecific(pthread_key_t key)`

### 数据的保护

 #### 1. Mutex互斥（Mutual Exclusion）

- 在共享数据访问的时候，去申请加把锁，谁先拿到锁，谁就拿到了访问权限，其他人就只好在门外等着，等这个人访问结束，把锁打开，其他人再去争夺，还是遵循谁先拿到谁访问。

- 一个转账的场景

  ```c
  
  #include <pthread.h>
  #include <stdio.h>
  #include <stdlib.h>
  
  #define NUM_OF_TASKS 5
  
  int money_of_tom = 100;
  int money_of_jerry = 100;
  //第一次运行去掉下面这行
  pthread_mutex_t g_money_lock;
  
  void *transfer(void *notused)
  {
    pthread_t tid = pthread_self();
    printf("Thread %u is transfering money!\n", (unsigned int)tid);
    //第一次运行去掉下面这行
    pthread_mutex_lock(&g_money_lock);
    sleep(rand()%10);
    money_of_tom+=10;
    sleep(rand()%10);
    money_of_jerry-=10;
    //第一次运行去掉下面这行
    pthread_mutex_unlock(&g_money_lock);
    printf("Thread %u finish transfering money!\n", (unsigned int)tid);
    pthread_exit((void *)0);
  }
  
  int main(int argc, char *argv[])
  {
    pthread_t threads[NUM_OF_TASKS];
    int rc;
    int t;
    //第一次运行去掉下面这行
    pthread_mutex_init(&g_money_lock, NULL);//初始化这个Mutex
  
    for(t=0;t<NUM_OF_TASKS;t++){
      rc = pthread_create(&threads[t], NULL, transfer, NULL);
      if (rc){
        printf("ERROR; return code from pthread_create() is %d\n", rc);
        exit(-1);
      }
    }
    
    for(t=0;t<100;t++){
      //第一次运行去掉下面这行
      pthread_mutex_lock(&g_money_lock);	//去抢这把锁，抢到了就执行下一行程序，对共享变量进行访问。没抢到，就被堵塞在这等待
      /*如果不想被阻塞，可以使用 pthread_mutex_trylock 去抢那把锁，如果抢到了，就可以执行下一行程序，对共享变量进行访问；如果没抢到，不会被阻塞，而是返回一个错误码。*/
      printf("money_of_tom + money_of_jerry = %d\n", money_of_tom + money_of_jerry);
      //第一次运行去掉下面这行
      pthread_mutex_unlock(&g_money_lock);//访问结束后，释放锁
    }
    //第一次运行去掉下面这行
    pthread_mutex_destroy(&g_money_lock);
    pthread_exit(NULL);
  }
  ```

  编译：

  `gcc mutex.c -lpthread`

#### 2. 条件变量

- 用pthread_mutex_trylock()，就可以不用等着，去干点儿别的，但是我怎么知道什么时候回来再试一下，是不是轮到我了呢？能不能在轮到我的时候，通知我一下呢？

  **通过条件变量。**

  但是当它接到了通知，来操作共享资源的时候，还是需要抢互斥锁，因为可能很多人都受到了通知，都来访问了，所以条件变量和互斥锁是配合使用的。

- 下面这个例子为什么要用条件变量，不单纯用互斥？

  - 如果不使用条件变量，而且BOSS也不是一直生产任务，那么这时互斥量就会空闲出来，总会有一个员工能拿到锁，员工线程这时候就会在while循环中不停的获得锁，判断状态，释放锁，这样的话就会十分消耗cpu资源了。
  - 这时候我们可能会想到，在while循环中加个睡眠，例如5秒，也就是员工线程每隔5秒来执行一次获得锁，判断状态，释放锁的操作，这样就会减轻cpu资源的消耗。
  - 但是实际应用场景中，我们无法判断到底间隔几秒来执行一次这个获得锁，判断状态，释放锁的流程，时间长了可能影响吞吐量，时间短了会造成cpu利用率过高，所以这时候引入了条件变量，将主动查询方式改成了被动通知方式，效率也就上去了。

- 分配3员工干活的例子

  ```c
  
  #include <pthread.h>
  #include <stdio.h>
  #include <stdlib.h>
  
  #define NUM_OF_TASKS 3
  #define MAX_TASK_QUEUE 11	//10 个任务，每个任务一个字符
  
  char tasklist[MAX_TASK_QUEUE]="ABCDEFGHIJ";
  int head = 0;	//当前分配的工作从哪里开始，如果 head 等于 tail，则当前的工作分配完毕；
  int tail = 0;	//当前分配的工作到哪里结束，如果 tail 加 N，就是新分配了 N 个工作。
  
  int quit = 0;
  
  pthread_mutex_t g_task_lock;//声明锁
  pthread_cond_t g_task_cv;	//声明条件变量
  
  void *coder(void *notused)
  {
    pthread_t tid = pthread_self();
  
    while(!quit){
  
      pthread_mutex_lock(&g_task_lock);
      //判断head和tail是否相等，不相等就是有任务，跳过while执行下面的任务去
      /*相等就是没任务调用 pthread_cond_wait 进行等待，这个函数会把锁也作为变量传进去。这是因为等待的过程中需要解锁，要不然，你不干活，等待睡大觉，还把门给锁了，别人也干不了活，而且老板也没办法获取锁来分配任务。*/
      while(tail == head){
        if(quit){
          pthread_mutex_unlock(&g_task_lock);
          pthread_exit((void *)0);
        }
        printf("No task now! Thread %u is waiting!\n", (unsigned int)tid);
          
        pthread_cond_wait(&g_task_cv, &g_task_lock);//通过互斥锁来阻塞线程
        printf("Have task now! Thread %u is grabing the task !\n", (unsigned int)tid);
      }
      char task = tasklist[head++];
      pthread_mutex_unlock(&g_task_lock);
      printf("Thread %u has a task %c now!\n", (unsigned int)tid, task);
      sleep(5);
      printf("Thread %u finish the task %c!\n", (unsigned int)tid, task);
    }
  
    pthread_exit((void *)0);
  }
  
  int main(int argc, char *argv[])
  {
    pthread_t threads[NUM_OF_TASKS];
    int rc;
    int t;
  
    pthread_mutex_init(&g_task_lock, NULL);	//初始化锁
    pthread_cond_init(&g_task_cv, NULL);		//初始化条件变量
  	
    //主线程上。它初始化了条件变量和锁，然后创建三个线程，也就是我们说的招聘了三个员工。
    for(t=0;t<NUM_OF_TASKS;t++){
      rc = pthread_create(&threads[t], NULL, coder, NULL);
      //一开始三个员工都是在等待的状态，因为初始化的时候，head 和 tail 相等都为零。  
      if (rc){
        printf("ERROR; return code from pthread_create() is %d\n", rc);
        exit(-1);
      }
    }
  
    sleep(5);
      
    /*接下来分配任务，总共 10 个任务。老板分四批分配，第一批一个任务三个人抢，第二批两个任务，第三批三个任务，正好每人抢到一个，第四批四个任务，可能有一个员工抢到两个任务。*/
    for(t=1;t<=4;t++){
      /*老板分配工作的时候，也是要先获取锁 pthread_mutex_lock，然后通过 tail 加一来分配任务，这个时候 head 和 tail 已经不一样了*/
      pthread_mutex_lock(&g_task_lock);
      tail+=t;
      printf("I am Boss, I assigned %d tasks, I notify all coders!\n", t);
      /*这个时候三个员工还在 pthread_cond_wait 那里睡着呢，接下来老板要调用 pthread_cond_broadcast 通知所有的员工，“来活了，醒醒，起来干活”*/
      pthread_cond_broadcast(&g_task_cv);	//通知（解锁）所有线程
      pthread_mutex_unlock(&g_task_lock);	//老板解锁
      /*3个员工醒过来先抢锁，抢锁由pthread_cond_wait 在收到通知的时候，自动做*/
      /*抢到锁的员工就通过 while 再次判断 head 和 tail 是否相同。这次因为有了任务，不相同了，所以就抢到了任务。而没有抢到任务的员工，由于抢锁失败，只好等待抢到任务的员工释放锁，抢到任务的员工在 tasklist 里面拿到任务后，将 head 加一，然后就释放锁。这个时候，另外两个员工才能从 pthread_cond_wait 中返回，然后也会再次通过 while 判断 head 和 tail 是否相同。不过已经晚了，任务都让人家抢走了，head 和 tail 又一样了，所以只好再次进入 pthread_cond_wait，接着等任务。*/  
      sleep(20);
    }
  
    pthread_mutex_lock(&g_task_lock);
    quit = 1;
    pthread_cond_broadcast(&g_task_cv);
    pthread_mutex_unlock(&g_task_lock);
  
    pthread_mutex_destroy(&g_task_lock);
    pthread_cond_destroy(&g_task_cv);		//销毁共享变量
    pthread_exit(NULL);
  }
  ```

### 写多线程的套路

  ![](https://static001.geekbang.org/resource/image/02/58/02a774d7c0f83bb69fec4662622d6d58.png)
