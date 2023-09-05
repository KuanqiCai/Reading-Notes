--------------------------------
[ERROR]:The imported target "Qt5::Gui" references the file "/usr/lib/x86_64-linux-gnu/libEGL.so" but this file does not exist. 
-------------------------------
```
~$ ls /usr/lib/x86_64-linux-gnu | grep -i libegl
~$ sudo rm /usr/lib/x86_64-linux-gnu/libEGL.so; sudo ln /usr/lib/x86_64-linux-gnu/libEGL.so.1 /usr/lib/x86_64-linux-gnu/libEGL.so
```

--------------------------
[ERROR]: cmake Could NOT find PythonInterp
---------------------------
将系统python的版本转成python3
```
基于update-alternatives命令。此方法为系统级修改。
直接执行下面两个命令即可：

~$ sudo update-alternatives --install /usr/bin/python python /usr/bin/python2 100
~$ sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 150

如果需要改回python2默认，则输入：

~$ sudo update-alternatives --config python

完毕。

```

-----------------------
[ERROR]: CMake Error: The following variables are used in this project, but they are set to NOTFOUND. Please set them or make sure they are set and tested correctly in the CMake files: OPENGL_gl_LIBRARY (ADVANCED)
---------------
```
~$ sudo rm /usr/lib/x86_64-linux-gnu/libGL.so
~$ sudo ln -s /usr/lib/x86_64-linux-gnu/mesa/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so
```

check the repositories:

```
sudo apt-get install \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev
```
