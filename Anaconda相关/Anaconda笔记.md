# 简介

Anaconda是一个环境管理软件，解决不同代码需要不同的环境时重复卸载安装环境的需求。

# 基本指令

- 查看版本号
  - `conda --version`
  - `conda -V`
  
- 更新Anaconda

  - 先更新conda:`conda update conda`
  - 在更新Anaconda: ``

- 查看当前系统下有什么conda环境
  - `conda env list`
  - `conda info --envs`
  
- 新建conda环境
  - `conda create -n [环境名]`
  
- 激活某一个环境
  - `conda activate environment`
  
- 退出该环境
  - `conda deactivate`
  
- 查看环境中所装的包
  - `conda list`

- 安装包

  - `conda install pkgs[==version]`不指定版本则下载最新版

  可以在激活环境时，直接制定python版本：

  - `conda create -n test_env python==3.8`

- 卸载包

  - `conda uninstall xx`

- conda安装本地软件包

  - `conda install --use-local package.tar.bz2`

# 环境相关的指令

- 克隆一个环境

  - `conda create -n new_env --clone old_env`
  
- 删除环境

  - `conda remove -n environment --all`
  
- 改名字：

  通过克隆的方式实现

  conda create --name python32（新名字） --clone python321（老名字）

  conda remove --name python321 --all

- 导出环境

  - `conda env export > environment.yml`
  
- 根据yml文件快速创建环境

  - `conda env create -f environment.yml`

- 用txt一键安装我们所需要的环境
  - 在激活某一环境后：`pip install -r requirements.txt`