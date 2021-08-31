# 信号阻塞sigprocmask()

- 有时候不希望在接到信号时就立即停止当前执行，去处理信号，同时也不希望忽略该信号，而是延时一段时间去调用信号处理函数。这种情况是通过阻塞信号实现的。

- 信号递达（Delivery）：执行信号的处理动作

  信号未决（Pending）：信号从产生到递达之间的状态

  - 进程可以选择阻塞（Block）某个信号。被阻塞的信号产生时将保持在未决状态，直到进程解除对此信号的阻塞，才执行递达的动作。

- sigprocmask()

  ```c
  #include <signal.h>      
  int sigprocmask(ubt how,const sigset_t*set,sigset_t *oldset); 
  ```

  **参数：**

  - how：用于指定信号修改的方式，可能选择有三种
    - SIG_BLOCK			
      - 将set所指向的信号集中包含的信号加到当前的信号掩码中。即信号掩码和set信号集进行或操作。
    - SIG_UNBLOCK
      - 将set所指向的信号集中包含的信号从当前的信号掩码中删除。即信号掩码和set进行与操作
    - SIG_SETMASK
      - 将set的值设定为新的进程信号掩码。即set对信号掩码进行了赋值操作
  - set：为指向信号集的指针，在此专指新设的信号集，如果仅想读取现在的屏蔽值，可将其置为NULL。
  - oldset：也是指向信号集的指针，在此存放原来的信号集。可用来检测信号掩码中存在什么信号。

  **返回：**

  - 成功执行时，返回0。失败返回-1，errno被设为EINVAL

- sigsuspend()

  ```c
  int sigsuspend(const sigset_t*sigmask);
  ```

  **进程执行到sigsuspend时，sigsuspend并不会立刻返回，进程处于TASK_INTERRUPTIBLE状态并立刻放弃CPU，等待UNBLOCK（mask之外的）信号的唤醒** 。进程在接收到UNBLOCK（mask之外）信号后，调用处理函数，然后把现在的信号集还原为原来的，sigsuspend返回，进程恢复执行。

- 例子：

  ```c
  #include <unistd.h>
  #include <signal.h>
  #include <stdio.h>
  void handler(int sig)   //信号处理程序
  {
     if(sig == SIGINT)
        printf("SIGINT sig");
     else if(sig == SIGQUIT)
        printf("SIGQUIT sig");
     else
        printf("SIGUSR1 sig");
  }
   
  int main()
  {
      sigset_t new,old,wait;   //三个信号集
      struct sigaction act;
      act.sa_handler = handler;
      sigemptyset(&act.sa_mask);
      act.sa_flags = 0;
      sigaction(SIGINT, &act, 0);    //可以捕捉以下三个信号：SIGINT/SIGQUIT/SIGUSR1
      sigaction(SIGQUIT, &act, 0);
      sigaction(SIGUSR1, &act, 0);
     
      sigemptyset(&new);
      sigaddset(&new, SIGINT);  //SIGINT信号加入到new信号集中
      sigemptyset(&wait);
      sigaddset(&wait, SIGUSR1);  //SIGUSR1信号加入wait
      sigprocmask(SIG_BLOCK, &new, &old);       //将SIGINT阻塞，保存当前信号集到old中
     
      //临界区代码执行    
    
      if(sigsuspend(&wait) != -1)  //程序在此处挂起；用wait信号集替换new信号集。即：过来SIGUSR1信  号，阻塞掉，程序继续挂起；过来其他信号，例如SIGINT，则会唤醒程序。执行sigsuspend的原子操作。注意：如果“sigaddset(&wait, SIGUSR1);”这句没有，则此处不会阻塞任何信号，即过来任何信号均会唤醒程序。
          printf("sigsuspend error");
      printf("After sigsuspend");
      sigprocmask(SIG_SETMASK, &old, NULL);
      return 0;
  }
  ```

  