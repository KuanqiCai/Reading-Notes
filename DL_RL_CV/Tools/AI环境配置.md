# 1.安装conda

1. 下载最新版本conda:https://docs.conda.io/en/latest/miniconda.html
2. 给予权限：`chmod u+x Miniconda3-latest-Linux-x86_64.sh `
3. 安装：`./Miniconda3-latest-Linux-x86_64.sh`

下面的安装都在conda环境中安装

# 2.安装Nvidia驱动

- 注意安装的版本

  要和下面的cudnn/cuda版本相匹配

  比如这里的driver:515+cuda:11.7+cudnn:8.5

- 两种方法安装驱动：

  1. "Software & Updates": Additional Drivers

     1. choose on driver to install automatically
     2. `reboot`

  2. install driver from official website

     1. diable Nouveau kernel driver

        - `sudo gedit /etc/modprobe.d/blacklist-nouveau.conf`

          add the following contents

          ```
          blacklist nouveau
          options nouveau modeset=0
          ```

        - `sudo update-initramfs -u`

        - `reboot`

     2. install the latest driver

        `https://www.nvidia.cn/Download/index.aspx?lang=cn#`

        - `sudo chmod +x NVIDIA-Linux-x86_64-515.76.run `

        - `sudo ./NVIDIA-Linux-x86_64-515.76.run `

# 3.安装cuda

- 注意版本要和nvidia匹配

- 从https://developer.nvidia.com/cuda-toolkit-archive选择对应的版本后得到下载路径：

  - `wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda_11.7.1_515.65.01_linux.run``

  - ``sudo sh cuda_11.7.1_515.65.01_linux.run`

  - add followings to .bashrc

    ```
    export PATH=$PATH:/usr/local/cuda/bin  
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64  
    export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/lib64
    ```

# 4.安装cudnn

- 注意版本要和nvidia和cuda匹配
- install pkg from https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
- `sudo dpkg -i cudnn-local-repo-ubuntu2004-8.5.0.96_1.0-1_amd64.deb `
- `sudo cp /var/cudnn-local-repo-ubuntu2004-8.5.0.96/cudnn-local-0579404E-keyring.gpg /usr/share/keyrings/`
- `sudo apt-get update`
- `sudo apt-get install libcudnn8=8.5.0.96-1+cuda11.7`
- `sudo apt-get install libcudnn8-dev=8.5.0.96-1+cuda11.7`
- `sudo apt-get install libcudnn8-samples=8.5.0.96-1+cuda11.7`

# 5.安装pytorch

- 根据官网得到安装命令：

  `conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`



# 6.多版本cuda切换

1. 安装第二个[cuda](https://developer.nvidia.com/cuda-toolkit-archive)版本时，选择runfile(local)安装

2. cudnn如果是11.x,12.x这样的大版本不同，也要安装对应的cudnn

3. 在.bashrc中选择使用哪个cuda:

   ```
   # cuda 12.1
   # export PATH="/usr/local/cuda-12.1/bin:$PATH"
   # export LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH"
   # export CUDA_HOME=/usr/local/cuda-12.1
   
   # cuda 11.8
   export PATH="/usr/local/cuda-11.8/bin:$PATH"
   export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"
   export CUDA_HOME=/usr/local/cuda-11.8
   ```

   