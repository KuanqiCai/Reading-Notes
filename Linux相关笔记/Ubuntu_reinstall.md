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
1.swap area,主分区，32G,空间起始位置
2. EFI分区，逻辑分区，500MB，空间起始位置
3.Ext 4 journal，/ ，主分区，剩下所有G，空间起始位置



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
