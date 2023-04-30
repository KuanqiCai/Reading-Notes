# 1. NeRF简介

## 1.1 什么是NeRF

- NeRF (Neural Radiance Fields)即神经辐射场，他用一个MLP神经网络去隐式地学习一个三维场景。
  - 输入：稀疏的多角度 且 有位姿的图像
  - 输出：神经辐射场模型（根据此模型可以渲染出任意角度下的照片）
  
- NeRF本质上是完成了3D渲染功能

  - 渲染：将场景定义（包括摄像机、灯光、表面几何和材料）转换为模拟摄像机图像的过程。

    简单来说就是模拟相机的拍照过程，生成的结果是该视角下看到的一张照片

  - 传统方法有光栅化（rasterization），光线追踪（ray tracing）

    相关课程：

    - [GAMES101 现代计算机图形学入门](https://www.bilibili.com/video/BV1X7411F744/?spm_id_from=333.337.search-card.all.click)

- 隐式表达和显式表达：

  - 隐式表达Implicit：告诉3D点满足特定的关系，而不是具体的点在哪里。
  
    例子：
  
    - 一个公式：$f(x,y,z)=0$
    - SDF
    - NeRF
  
  - 显示表达Explicit：直接给出3D点或3D点的参数映射
  
    例子：
  
    - 点云
    - 网格
    - 体素

## 1.2 NeRF源码安装

- [NeRF论文](https://www.matthewtancik.com/nerf)是用TensorFlow做的：[TensorFlow版本](https://github.com/bmild/nerf)

- 使用别人写好的[Pytorch版本](https://github.com/yenchenlin/nerf-pytorch)的NeRF

  ```
  git clone https://github.com/yenchenlin/nerf-pytorch.git
  cd nerf-pytorch
  pip install -r requirements.txt
  ```

- 如何运行见[Pytorch版本](https://github.com/yenchenlin/nerf-pytorch)的README。

## 1.3 Instant-ngp安装

[Instant-ngp](https://github.com/NVlabs/instant-ngp)使用了[Tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)5s算完NeRF,否则要算五个半小时

```
$ git clone --recursive https://github.com/nvlabs/instant-ngp
$ cd instant-ngp
$ cmake . -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc -DTCNN_CUDA_ARCHITECTURES=61
$ cmake --build build --config RelWithDebInfo -j
```

使用见git[主页](https://github.com/NVlabs/instant-ngp#usage)