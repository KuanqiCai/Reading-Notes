资料：

- [blog](https://www.liaoxuefeng.com/wiki/1016959663602400/1019273143120480)
- [documentation](https://docs.python.org/3/library/venv.html)

# 1. Venv

## 1.1 为什么用

在开发Python应用程序的时候，系统安装的Python3只有一个版本：3.10。所有第三方的包都会被`pip`安装到Python3的`site-packages`目录下。

如果我们要同时开发多个应用程序，那这些应用程序都会共用一个Python，就是安装在系统的Python 3。如果应用A需要jinja 2.7，而应用B需要jinja 2.6怎么办？

这种情况下，每个应用可能需要各自拥有一套“独立”的Python运行环境。venv就是用来为一个应用创建一套“隔离”的Python运行环境。

## 1.2 基本使用：

假设开发新项目project01

1. 在project01下(其实任意地方都行)创建目录:project01env

   ```
   $ mkdir project01env
   $ cd project01env/
   ```

2. 创建独立的python运行环境

   ```shell
   $ python3 -m venv .
   $ ls
   bin  include  lib  lib64  pyvenv.cfg  share
   ```

   - `python3 -m venv <目录>`：创建了一个独立的python运行环境

     比如：`python3 -m venv .env`

3. 激活虚拟环境

   ```shell
   $ cd bin
   $ ls
   activate      activate.fish  easy_install      pip   pip3.8  python3
   activate.csh  Activate.ps1   easy_install-3.8  pip3  python
   # linux
   $ source activate
   (project01env) xuy1fe:~/Desktop/project01env/bin$ 

   # windows
   ## powershell:
   PS C:\Users\51212\Desktop\gym-examples\.env\Scripts> .\Activate.ps1
   ## cmd:
   PS C:\Users\51212\Desktop\gym-examples\.env\Scripts> .\activate.bat
   ```

   - `bin`目录下有python3, pip3等可执行文件，实际上是链接到Python系统目录的软链接。
   - 激活后，命令行前面就会出现环境名，即pyvenv.cfg所在的目录
   - 在`venv`环境下，用pip安装的包，都安装在 `/project01env/lib/python3.x/site-packages`

4. 退出环境

   ```
   $ deactivatpipe
   ```

5. 如果不再需要这个虚拟环境

   直接删除project01env文件夹。