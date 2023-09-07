### 1.安装realtime kernel （https://www.cnblogs.com/Pyrokine/p/16695196.html）

1. 查看当前内核版本

```c++
uname -r
```

2. 笔者当前版本为 5.15.0-46-generic ，去下面网址中下载版本相近的内核文件和补丁文件，笔者选择的是 linux-5.15.65.tar.gz 和 patch-5.15.65-rt49.patch.gz ，可以先下 patch 包，然后根据对应的版本找内核包，不过不一定需要版本完全一致，相近的版本大概率也是可行的

```c++
https://www.kernel.org/pub/linux/kernel/
https://www.kernel.org/pub/linux/kernel/projects/rt/
```

3. 解压内核然后解压补丁，版本不同的话可以使用 TAB 键来补全

```c++
tar -xzvf linux-5.15.65.tar.gz
cd linux-5.15.65
gzip -cd ../patch-5.15.65-rt49.patch.gz | patch -p1 --verbose
```

4. 安装依赖

```c++
sudo apt-get install autoconf bison build-essential dkms dwarves fakeroot flex libelf-dev libiberty-dev libidn11 libidn11-dev libncurses-dev libpci-dev libssl-dev libudev-dev minizip openssl zlibc zstd
```

5.复制当前内核的配置文件（用 TAB 补全），进入 config 交互界面调整下面设置，SAVE 后 EXIT

```c++
cp /boot/config-xxxx-generic ~/.config
make menuconfig
```

- General Setup -> Preemption Model 设置为 Fully Preemptible Kernel(RT)
- General Setup -> Timers subsystem -> Timer tick handling 设置为 Full dynticks system
- General Setup -> Timers subsystem 开启 High Resolution Timer Support
- Processor type and features -> Timer frequency 设置为 1000 HZ

6.然后编辑 .config 并修改以下内容

```c++
sudo gedit .config
```

- CONFIG_SYSTEM_TRUSTED_KEY=""
- CONFIG_SYSTEM_REVOCATION_KEYS=""

7. 编译内核，线程数设为和 CPU 线程数相同

```c++
make -j`nproc` deb-pkg
```

8. 安装内核，* 不要替换，所有编译后的文件都需要安装

```c++
sudo dpkg -i ../*.deb
```

9. 查看当前所有已安装内核

```c++
cat /boot/grub/grub.cfg | grep "menuentry 'Ubuntu"
```

10. 设置开机等待时间手动选择内核，默认以新内核启动，高级选项里面有旧内核

```c++
sudo gedit /etc/default/grub
# 修改下面键值
GRUB_TIMEOUT_STYLE=menu
GRUB_TIMEOUT=5
```

11. 更新grub后重启查看下内核是否更新了

```c++
sudo update-grub
uname -r
```

### 安装 Nvidia 显卡驱动

1. 首先重启切回非实时内核，然后禁用nouveau，在文件末尾插入以下内容

```c++
sudo gedit /etc/modprobe.d/blacklist.conf
```

- blacklist nouveau
- options nouveau modeset=0

2. 重启后验证，如无任何输出则生效

```c++
lsmod | grep nouveau
```

3. 卸载显卡驱动

```c++
sudo apt purge nvidia*
```

4. 去 Nvidia 官网下载安装文件后安装（即不能直接安装二进制版本），在有 DKMS 字样的页面选择 YES ，其他都选择 NO ，否则可能无法开机，因此先设定好 root 密码再重启

```c++
chmod +x xxx.run
sudo bash ./xxx.run
```

5. 查看是否安装成功

```c++
nvidia-smi
```

6. 测试实时性，观察最右侧的 MAX ，如果一直只有几十，说明是实时系统了，如果是非实时系统，运行一段时间会增加到几千甚至上万

```c++
sudo apt-get install rt-tests
sudo cyclictest -a -t -p 99
```

### Franka_ros (https://frankaemika.github.io/docs/installation_linux.html#installing-from-the-ros-repositories)

1. Building libfranka

```c++
sudo apt install build-essential cmake git libpoco-dev libeigen3-dev
```

2. Then, download the source code by cloning libfranka from GitHub (https://github.com/frankaemika/libfranka).

3. For Panda you need to clone:

```c++
git clone --recursive https://github.com/frankaemika/libfranka # only for panda
cd libfranka
```

4. In the source directory, create a build directory and run CMake:

```c++
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF ..
cmake --build .
```

5. Install

```c++
sudo make install
```

6. Building the ROS packages

After setting up ROS Noetic, create a Catkin workspace in a directory of your choice

Then clone the franka_ros repository from GitHub:

```c++
git clone --recursive https://github.com/frankaemika/franka_ros src/franka_ros
```

### Allow a user to set real-time permissions for its processes

1. After the real time kernel is installed and running, add a group named realtime and add the user controlling your robot to this group:

```c++
sudo addgroup realtime
sudo usermod -a -G realtime $(whoami e.g. ckq)
reboot
groups // check whether realtime group is added
```

2. Afterwards, add the following limits to the realtime group in /etc/security/limits.conf:

- @realtime soft rtprio 99
- @realtime soft priority 99
- @realtime soft memlock 102400
- @realtime hard rtprio 99
- @realtime hard priority 99
- @realtime hard memlock 102400

The limits will be applied after you log out and in again.
