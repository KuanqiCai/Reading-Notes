binary是编译好的可以直接使用，  
source是还没编译过的源代码，需要自行编译。

- 一、下载安装包后安装 
  Ubuntu用的安装包是deb，CentOS用的是rpm

  ```
  ~$ sudo rpm -i jdk_xxx_linux_x64_bin.rpm     //CentOS安装命令
  ~$ sudo dpkg -i jdk_xxx_linux_x64_bin.deb   // Ubuntu安装命令。-i 就是install的意思
  ```

  - `dpkg -l | grep jdk`

    - dpkg -l 显式所有安装的软件列表

    - | 是管道，dpkg -l的输出作为grep jdk的输入
    - grep命令 搜索关键词jdk

  - `dpkg -r`    删除软件

- 二、用软件管家：apt-get来安装
  - 安装：`apt-get install openjdk-9-jdk`
  - 卸载：` apt-get purge openjdk-9-jdk `
  - 配置文件在 /etc/apt/sources.list里，记录了apt从哪个服务器下载这些软件。
  - Linux中的软件
    - 主执行文件会放在 /usr/bin 或者 /usr/sbin 下面
    - 其他的库文件会放在 /var 下面
    - 配置文件会放在 /etc 下面

- 三、解压缩安装

  Linux长用tar.gz格式的压缩包

  - `tar xvzf jdk-XXX_linux-x64_bin.tar.gz`解压缩

  - `export PATH = $JAVA_HOME/bin:$PATH`配置环境变量

    export命令仅在当前命令行的会话中管用，一旦退出重新登录进来，就不管用了

  - `vim .bashrc`配置永远管用

    在当前用户的默认工作目录，例如 /root 或者 /home/cliu8 下面，有一个隐藏的.bashrc 文件，用ls -la可看到。在.bashrc文件中加入export的配置后，就一直有用了。

- 安装gcc
```
~$ sudo apt install build-essential      //安装所有c/c++编译所需要的包
~$ g++ --version     //查看当前g++版本
```
- 安装 cmake
```
~$ sudo apt-get remove cmake     //卸载原版本
自https://cmake.org/download/下载cmake源码，如cmake-3.20.0 tar.gz
~$ tar -zxv -f cmake-3.17.1.tar.gz   //在cmake源码所在文件夹中打开命令终端，解压文件
~$ cd cmake-3.20.0-rc3
~$ ./bootstrap
~$ make            //编译构建
~$ sudo make install  //安装
~$ cmake --version //查看版本
```
- 安装vim
```
sudo apt-get install vim-gtk
```