# 一、基础

## 1. 安装orb-slam2

1. 环境

   ```
   openCV4
   Pangolin
   ROs noteic
   ubuntu 20
   ```

2. 下载orb-slam2后修改

   1. ORB_SLAM2/include/ORBextractor.h:`#include <opencv2/opencv.hpp>`代替`include <opencv/cv.h>`

   2. ORB_SLAM2文件夹下的CMakeList.txt:

      1. `OpenCV 3`改为`OpenCV 4`
      2. `find_package(Eigen3 3.1.0 REQUIRED)` to `find_package(Eigen3 3.1.0 REQUIRED NO_MODULE)`

   3. ORB_SLAM2/include/system.h:添加`#include<unistd.h>`

   4. ORB_SLAM2/include/LoopClosing.h:

      ```
      typedef map<KeyFrame*,g2o::Sim3,std::less<KeyFrame*>,
              Eigen::aligned_allocator<std::pair<const KeyFrame*, g2o::Sim3> > > KeyFrameAndPose;
      //改为：
      typedef map<KeyFrame*,g2o::Sim3,std::less<KeyFrame*>,
              Eigen::aligned_allocator<std::pair<KeyFrame *const, g2o::Sim3> > > KeyFrameAndPose;
      
      ```

      

      