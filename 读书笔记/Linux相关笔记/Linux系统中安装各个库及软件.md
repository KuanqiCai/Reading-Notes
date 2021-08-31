binary是编译好的可以直接使用，  
source是还没编译过的源代码，需要自行编译。

- 下载安装包后安装 
Ubuntu用的安装包是deb，CentOS用的是rpm
```
~$ sudo rpm -i jdk_xxx_linux_x64_bin.rpm     //CentOS安装命令
~$ sudo dpkg -i jdk_xxx_linux_x64_bin.deb   // Ubuntu安装命令。-i 就是install的意思
```

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