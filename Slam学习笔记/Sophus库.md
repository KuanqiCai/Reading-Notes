# 安装

1. 安装依赖库fmt

   ```shell
   # 最新版的fmt会导致Sophus编译报错，所以安装8.0.0版本
   # 首先删除之前安装的版本，一般make install完了目录下都会有一个install_manifest.txt的文件记录安装的所有内容，通过如下命令来删除：
   xargs rm < install_manifest.txt
   git clone -b 8.0.0 git@github.com:fmtlib/fmt.git
   cd fmt
   mkdir build 
   cd build
   cmake ..
   make
   sudo make install
   ```

2. 安装Sophus库

   ```shell
   git clone git@github.com:strasdat/Sophus.git
   cd Sophus
   mkdir build
   cd build
   cmake ..
   make
   sudo make install
   ```

   