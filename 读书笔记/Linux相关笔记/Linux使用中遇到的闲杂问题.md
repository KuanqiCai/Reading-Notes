- 升级ubuntu版本时遇到的问题
   - 如何升级
   ```
   ~$ update-manager -c -d 
   ```
   - ubuntu升级时磁盘空间 /boot 不足
      1. ~$ df -h：查看磁盘存储情况
      2. ~$ uname -a： 查看当前使用内核版本
      3. ~$ sudo apt-get remove linux-image-： 查看所有的内核版本
      4. ~$ sudo apt-get remove linux-image-4.18.0-25-generic：删除对应版本
- 权限问题
   - /opt相当于D:/software.但没法创创建新文件夹
   ```
   ~$ sudo chmod 777 /opt
   ```
- 帮助文档  
  以mkdir为例，2种方法
   - mkdir --help
   - man mkdir
  
- ubuntu 18.04没法全屏
   - sudo apt-get update
   - sudo apt-get install open-vm-tools