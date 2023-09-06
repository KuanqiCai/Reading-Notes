# 双系统安装

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
1.swap area,主分区，32G,空间起始位置
2. EFI分区，逻辑分区，500MB，空间起始位置
3. / ，主分区，剩下所有G，空间起始位置



## 3. 安装pinyin输入

安装fcitx-googlepinyin

```
sudo apt-get install fcitx-googlepinyin
```

配置language support， 从（6）开始

https://blog.csdn.net/qq_20016593/article/details/127563380
