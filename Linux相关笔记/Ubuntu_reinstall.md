# 双系统安装 ubuntu20.04

## 1. 删除ubuntu系统

1. 删除ubuntu所在卷

进入windows系统，右键此电脑-管理-磁盘管理-删除ubuntu所在的卷 (右键所在卷-删除)

遇到EFI无法右键删除：
 https://zhuanlan.zhihu.com/p/561777532

如果重启后出现：
GNU GRUB version 2.06
Minimal Bash-lke line editing is supported.  TAB lists possible comand completions, Anywhere else TAB 1ists possible device or flle completions.

则按照上述链接将window下面的EFI中的ubuntu删去，即assign 字母后，右键记事本，通过管理员权限进入，点击文件，点击打开，找到赋值字母的磁盘卷，打开后删去EFI内的ubuntu文件

## 2. 安装ubuntu系统

https://blog.csdn.net/YIBO0408/article/details/123937450

分区：
-1.swap area,主分区，32G,空间起始位置

-2. EFI分区，逻辑分区，500MB，空间起始位置

-3.Ext 4 journal，/ ，主分区，剩下所有G，空间起始位置



## 3. 安装pinyin输入

- Open Settings, go to Region & Language -> Manage Installed Languages -> Install / Remove languages.-
- Select Chinese (Simplified). Make sure Keyboard Input method system has Ibus selected. Apply.
- Reboot
- Log back in, reopen Settings, go to Keyboard.
- Click on the "+" sign under Input sources.
- Select Chinese (China) and then Chinese (Intelligent Pinyin).

# 安装基础软件

## 1. Terminator

terminator:`sudo apt-get install terminator`

美化：`sudo gedit ~/.config/terminator/config`并输入 (ps. 如果没有terminator文件，则需要自己建立一个)

```
[global_config]
  handle_size = -3
  title_transmit_fg_color = "#000000"
  title_transmit_bg_color = "#3e3838"
  inactive_color_offset = 1.0
  enabled_plugins = CustomCommandsMenu, LaunchpadCodeURLHandler, APTURLHandler, LaunchpadBugURLHandler
  suppress_multiple_term_dialog = True
[keybindings]
[profiles]
  [[default]]
    background_color = "#2e3436"
    background_darkness = 0.8
    background_type = transparent
    cursor_shape = ibeam
    cursor_color = "#e8e8e8"
    font = Ubuntu Mono 14
    foreground_color = "#e8e8e8"
    show_titlebar = False
    scroll_background = False
    scrollback_lines = 3000
    palette = "#292424:#5a8e1c:#00ff00:#cdcd00:#1e90ff:#cd00cd:#00cdcd:#d6d9d4:#4c4c4c:#868e09:#00ff00:#ffff00:#4682b4:#ff00ff:#00ffff:#ffffff"
    use_system_font = False
[layouts]
  [[default]]
    [[[child1]]]
      parent = window0
      profile = default
      type = Terminal
    [[[window0]]]
      parent = ""
      size = 925, 570
      type = Window
[plugins]
```
## 2.python3.9 (不一定需要)
http://www.taodudu.cc/news/show-5381101.html?action=onClick

## pip install
https://stackoverflow.com/questions/65644782/how-to-install-pip-for-python-3-9-on-ubuntu-20-04
## solve terminal cannot open
https://blog.csdn.net/weixin_46584887/article/details/120702843

## 3. BUG: Ubuntu每次启动都显示System program problem detected

解决方案：
```
sudo gedit /etc/default/apport
```

enabled=1 修改为 enabled=0

保存退出重启就不会提示了

## 4. 安装ros

https://blog.csdn.net/weixin_44244190/article/details/126854911

## 5. 安装 opencv (ubuntu 20.04 + opencv 4.5)

https://blog.csdn.net/weixin_44796670/article/details/115900538

https://blog.csdn.net/bj233/article/details/113351023

1. 安装依赖

①安装g++, cmake, make, wget, unzip，若已安装，此步跳过

```
sudo apt install -y g++
sudo apt install -y cmake
sudo apt install -y make
sudo apt install -y wget unzip
```

②安装opencv依赖的库

```
sudo apt-get install build-essential libgtk2.0-dev libgtk-3-dev libavcodec-dev libavformat-dev libjpeg-dev libswscale-dev libtiff5-dev
```

③安装一些可选的库

```
# python3支持（首次安装了python的库，但make报错了，之后删了这两个库，若不使用python，建议不安装）
sudo apt install python3-dev python3-numpy
# streamer支持
sudo apt install libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev
# 可选的依赖
sudo apt install libpng-dev libopenexr-dev libtiff-dev libwebp-dev
```

2. 下载OpenCV 4.5.0源文件

可以在官网下载相应版本的OpenCV，主要有Source和GitHub两种方式下载。

- Opencv：https://opencv.org/releases/ （点击source下载） 

- Opencv_contrib: https://github.com/opencv/opencv_contrib/tree/4.8.0 在Tags里找到相应版本下载，注意版本要与opencv一致

- 下载好解压后，将opencv_contrib3.4.13放在opencv3.4.13文件夹里面（为方便后续操作，可将上面两个文件夹分别命名为opencv和opencv_conrib）

3. 安装

- 在opencv文件夹下新建build文件夹

```
cd opencv
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules -D OPENCV_GENERATE_PKGCONFIG=YES ..
//后面“../opencv_contrib-3.4.0/modules”为你opencv_contrib的modules文件夹所在的路径
// -D OPENCV_GENERATE_PKGCONFIG=YES; OpenCV4以上默认不使用pkg-config，该编译选项开启生成opencv4.pc文件，支持pkg-config功能

make -j24

sudo make install
```
4. 环境配置
4.1 配置pkg-config环境

opencv4.pc文件的默认路径：/usr/local/lib/pkgconfig/opencv4.pc
若此目录下没有，可以使用以下命令搜索：

```
sudo find / -iname opencv4.pc
```

可以看到的确在这个目录下:

- /home/ckq/opencv-4.8.0/build/unix-install/opencv4.pc
- /usr/lib/x86_64-linux-gnu/pkgconfig/opencv4.pc

将路径加入到PKG_CONFIG_PATH（用gedit打开）：

```
sudo vim /etc/bash.bashrc
```

打开文件后在末尾输入：

```
PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig
export PKG_CONFIG_PATH

```

保存并退出后激活：

```
# 激活
source /etc/bash.bashrc

```

用以下命令验证是否成功：

```
pkg-config --libs opencv4
```

-lopencv_stitching -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_dnn_objdetect -lopencv_dnn_superres -lopencv_dpm -lopencv_highgui -lopencv_face -lopencv_freetype -lopencv_fuzzy -lopencv_hdf -lopencv_hfs -lopencv_img_hash -lopencv_line_descriptor -lopencv_quality -lopencv_reg -lopencv_rgbd -lopencv_saliency -lopencv_shape -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrapping -lopencv_superres -lopencv_optflow -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_dnn -lopencv_plot -lopencv_ml -lopencv_videostab -lopencv_videoio -lopencv_viz -lopencv_ximgproc -lopencv_video -lopencv_xobjdetect -lopencv_objdetect -lopencv_calib3d -lopencv_imgcodecs -lopencv_features2d -lopencv_flann -lopencv_xphoto -lopencv_photo -lopencv_imgproc -lopencv_core


4.2 配置动态库环境

① 打开文件（可能为空文件）：

```
sudo gedit /etc/ld.so.conf.d/opencv4.conf
```
② 在该文件末尾加上OpenCV的lib路径，保存退出：

```
/usr/local/lib
```

③ 使配置的路径生效：

```
sudo ldconfig
```


5. 测试OpenCV

cd 到/opencv/samples/cpp/example_cmake目录下，依次执行以下命令：

```
cmake .
make
./opencv_example
```

该测试需要电脑有摄像头，若启动摄像头看到了画面，说明安装成功


## 6. 安装google浏览器

https://blog.csdn.net/xyywendy/article/details/124342058

## 7. 安装openrave - fast IK solution

https://robots.uc3m.es/installation-guides/install-openrave.html#install-openrave-via-scripts-ubuntu-1804-bionic-and-ubuntu-2004-focal

Install OpenRAVE via scripts (Ubuntu 18.04 Bionic and Ubuntu 20.04 Focal)
Tested and works on fresh installs. Easy, but not guaranteed to work, nor to be the fastest mechanism (e.g. fcl not mandatory, and osg could alternatively be installed via apt in 20.04 Focal). Provides:

- Ubuntu 20.04 Focal: OpenRAVE 0.54.0 with Python 2 bindings, FCL, and OpenSceneGraph viewer.
- Ubuntu 18.04 Bionic: OpenRAVE 0.9.0 with Python 2 bindings, FCL, and Qtcoin viewer.

```
sudo apt install git lsb-release # probably already installed
```

On a fresh 20.04 Focal had to configure git email and user, even dummy okay:

```
git config --global user.name "FIRST_NAME LAST_NAME"
git config --global user.email "MY_NAME@example.com"
```

Always pay attention to prompts for sudo (and insert password):

```
cd  # go home
git clone https://github.com/crigroup/openrave-installation
cd openrave-installation
./install-dependencies.sh
./install-osg.sh
./install-fcl.sh
./install-openrave.sh
```


## 8.ubuntu + eigen3 安装（解决 fatal error: Eigen/Core: No such file or directory）

１．安装
```
sudo apt-get install libeigen3-dev
```
2. 解决 fatal error: Eigen/Core: No such file or directory

当调用 eigen 库时，会报错：fatal error: Eigen/Core: No such file or directory

这是因为 eigen 库默认安装在了 /usr/include/eigen3/Eigen 路径下，需使用下面命令映射到 /usr/include 路径下
```
sudo ln -s /usr/include/eigen3/Eigen /usr/include/Eigen
```

## 9.Matlab

1. 下载地址：

https://ww2.mathworks.cn/downloads/

2. 解压后 “sudo ./install”安装
   
3. When given the option to 'Create symbolick links to MATLAB scripts in:' it is recommended that you check the box and use the supplied path /usr/local/bin.

4. 创建快捷方式：

   创建matlab.desktop文件：

    在终端中输入以下代码（CTRL+C复制，CTRL+SHIFT+V粘贴），并按下ENTER键。输入登录密码（不会显示密码），并按下ENTER键。
```
sudo gedit /usr/share/applications/matlab.desktop
```

2、配置文件：

    在打开的文本文件中复制并粘贴以下文字，点击保存：
```
[Desktop Entry]

Version=1.0

Type=Application

Terminal=false

Exec=/usr/local/MATLAB/R2023a/bin/matlab -desktop

Name=MATLAB

Icon=/usr/local/MATLAB/R2023a/bin/glnxa64/cef_resources/matlab_icon.png

Categories=Math;Science

Comment=Scientific computing environment

StartupNotify=true

StartupWMClass=com-mathworks-util-PostVMInit
```

注：Exec的值是MATLAB执行文件，Icon的值是MATLAB应用程序图标的位置。如果安装matlabR2023a时未使用默认安装路径，请自行调整路径。 

Bug: com.jogamp.opengl.GLException: X11GLXDrawableFactory - Could not initialize shared resources for X11GraphicsDevice[type .x11, connection :1, unitID 0, handle 0x0, owner false, ResourceToolkitLock[obj 0x7c1bc33e, isOwner false, <2af5b1a, 46860956>[count 0, qsz 0, owner <NULL>]]] at jogamp.opengl.x11.glx.X11GLXDrawableFactory$SharedResourceImple

打开 matlab在command line输入：
```
opengl('save','software')
```
重启matlab

## 10.Visual Studio Code

1. Install link: https://code.visualstudio.com/docs/setup/linux (.deb package (64-bit))
   
2. sudo apt install ./<file>.deb

3. https://blog.csdn.net/qq_27386899/article/details/121455952 (vsccode 配置)

## 11. 安装Vrep

下载网址:
       https://www.coppeliarobotics.com/downloads

下载后，找到下载的目录，执行命令解压（或者直接右击下载的压缩包---Extract Here）

tar -xf CoppeliaSim_Edu_V4_2_0_Ubuntu20_04.tar.xz
（注：目前应该已经更新到了4.4版本，以后应该还会更新，这个版本问题根据需要选择就可）

添加环境变量。

在Home中打开.bashrc文件，添加（要根据自己的实际文件路径）

export COPPELIASIM_ROOT=/home/ckq/Downloads/CoppeliaSim_Edu_V4_2_0_rev0_Ubuntu20_04
运行，在解压好的文件中，打开终端，运行

xxx@xxx:~/Downloads/CoppeliaSim_Edu_V4_2_0_Ubuntu20_04$ ./coppeliaSim.sh

## 12. Kazam 录屏

sudo apt-get install kazam

## 13. 下载 Edge Dev

下载网站： https://www.microsoft.com/en-us/edge/download/insider?form=MA13FJ

下载后在安装包所在的位置打开terminal: 
```
sudo dpkg -i microsoft-edge-dev_88.0.673.0-1_amd64.deb
```

## 壁纸

推荐⼏个下载 4K 8K 超⾼清壁纸的⽹站：

https://pixabay.com

https://unsplash.com

https://wallpapersite.com

https://wallpapershome.com
