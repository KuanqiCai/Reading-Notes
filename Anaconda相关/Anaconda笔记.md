# 简介

Anaconda是一个环境管理软件，解决不同代码需要不同的环境时重复卸载安装环境的需求。

# 安装与卸载

## 1. 安装

- 下载包：https://docs.conda.io/en/latest/miniconda.html
- 双击运行下载(不添加环境变量)

## 2. 卸载

- 直接app and features中卸载



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

- 升级包

  - `conda update xx`

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

# Jupyter相关

## 1. 安装

### 1.1jupyter安装

`conda install jupyter`

### 1.2插件安装

https://github.com/ipython-contrib/jupyter_contrib_nbextensions
第一步：conda install -c conda-forge jupyter_contrib_nbextensions
第二步：jupyter contrib nbextension install --user
第三步：conda install -c conda-forge jupyter_nbextensions_configurator

## 2. 删除核

1. 查看所有核心:

   ```
   jupyter kernelspec list
   ```

2. 卸载指定核心

   ```
   jupyter kernelspec remove tensorflow-gpu
   ```

   