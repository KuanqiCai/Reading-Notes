githubt官方文档：https://docs.github.com/cn

# 一、基本配置

- 配置Git
  
   这里name和email不一致，会使得github小绿点不更新
   
   - ~$ git config --global user.name "Fernweh-yang"
     - TUM GITLAB: git config --global user.name "Yang Xu"
   - ~$ git config --global user.email "512127058@qq.com"
     - TUM Gitlab: git config --global user.email "ge23ged@mytum.de"
   - ~$ git config --list   //查看配置信息
   
- 获得帮助，以config 为例
   1. git help config
   2. git config --help
   3. man git-config
   
- 如何配置ssh到github
   - https://docs.github.com/cn/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent
   
   
   Windows:
   - 创建私钥
   ```
   ssh-keygen -t ed25519 -C "邮箱地址"
   ```
   - 复制ssh私钥
   ```
   cat ~/.ssh/id_ed25519.pub | clip
   ```
   Linux:
   - 检查现有SSH密钥
   ```
   ls -al ~/.ssh
   ```
   - 生成新的ssh
   ```
   $ ssh-keygen -t ed25519 -C "your_email@example.com"
   ```
   - 复制ssh
   ```
   $ sudo apt-get update
   $ sudo apt-get install xclip
   $ xclip -sel clip < ~/.ssh/id_ed25519.pub
   ```
   
- windows下更新git 版本
   - git update-git-for-windows

# 二、基本配置

## 2.1 通过git bash 上传文件

1. ```git init```:  在该文件夹下生成一个.git隐藏文件，初始化一个Git仓库。
2. ```git add``` . : 将文件夹下所有的文件添加到缓存区stage,将.换成文件名就是提交该文件
3. ```git status``` :查看现在的状态
4. `git commit -m "这里使注释"`: 把缓存区内所有的由add添加来的文件提交到分支。
5. `git remote add origin https://github.com/Fernweh-yang/TUM-Assigments`: 添加新的git方式origin， github 上创建好的仓库和本地仓库进行关联
6. `git push origin main`: 将本地库的所有内容推送到远程仓库的main分支上.
    - `git push -u origin master`: 第一次推送master分支时，加上-u参数，git会在推送内容的同时，关联本地master和远程master分支，以后可以简化命令。
7. `git diff 文件名` : 查看上次修改了什么内容
8. `git log` : 查看我们所有的commit记录

## 2.2 其他地方更改了库，本地更新

```shell
$ git remote add origin [//your github url]

//pull those changes

//origin就是一个名字，它是在你clone一个托管在Github上代码库时，git为你默认创建的指向这个远程代码库的标签，
$ git pull origin main 

// or optionally, 'git pull origin master --allow-unrelated-histories' if you have initialized repo in github and also committed locally

//now, push your work to your new repo

$ git push origin main
```

也可以：git pull = git fetch + git merge

```shell
$ git remote add upstream https://github.com/yeasy/docker_practice
$ git fetch upstream
$ git rebase upstream/main 
或者$ git merge upstream
```

[git merge和git fetch的区别](https://www.jianshu.com/p/f23f72251abc)

## 2.3 退回某一个版本

- 在git中，用HEAD表示当前版本也就是最新的commit id，上一个版本使HEAD^,上上个使HEAD^^,还可以用HEAD~100表示往上100个版本。HEAD指向的是分支，如master指向最新的提交，HEAD指向master就确定了当前分支的提交点。
1. `git reset --hard HEAD^` : 退回到上一个版本
2. `git reset --hard 版本号` : 回到指定的版本,也可以在后悔回退到过去时由此退回到未来
3. `git reflog`: 查看git记录下的每一条命令。如果忘了之前的版本号，由此查看。

- 撤销删改
   1. `git checkout -- file`: 撤销文件在工作区的全部修改，回到最近一个git commit 或git add时的状态。

   2. `git reset HEAD 文件名`: 如果add到缓存区了，可以用这个名利把缓存区的修改撤销掉unstage,重新回到工作区 

   3. `git rm 文件名` 从版本库中删除文件，和add一样之后要跟commit。

   4. `git remote -v`:查看远程库的信息

   5. `git remote rm 库名`:删除本地与远程库的绑定关系

## 2.4 分支管理

- `git checkout -b dev`: 创建并切换到dev分支,相当于下面2条命令。

- `git branch dev`
- `git checkout dev`
    2. `git branch`:查看当前分支
    3. `git merge xx`: 将分支xx合并到当前分支。
    4. `git branch -d dev` : 删除当前分支
    5. `git switch -c dev`:创建并切换到dev分支，和checkout的一个功能一致但因为checkout还有其他作用，所以switch更易理解。
    6. `git switch master`: 切换到master分支 
    7. `git log --graph`:可以看到分支合并图
    8. `git stash`: 将当前工作现场储存起来，等以后恢复现场继续工作
       1. `git stash list` 查看储存起来的工作现场
       2. `git stash pop`: 恢复工作现场，并删除stash内容
       3. `git stash apply`: 恢复工作现场，不删除stash内容，得输入`git stash drop`来删除。
    9. `git cherry-pick <commit id>`: 在master分支上修复了bug，而dev其他工作开发了一半，显然不可能从master重新分出来，在dev上再修复bug很麻烦。该命令可以把修复bug的那一次修改复制到dev分支来。

## 2.5 多人协作

1. 查看远程库信息，使用`git remote -v`；
2. 首先，试图用`git push origin <branch-name>`推送自己的修改
3. 如果推送失败，是因为远程分支比本地的更新，需要先用`git pull`试图合并
4. 如果合并有冲突，则解决冲突，没有冲突或者解决掉冲突后，再用`git push origin <branch-name>`推送就能成功
- 如果`git pull`提示`no tracking information`，则说明本地分支和远程分支的链接关系没有创建
    - 在本地创建和远程分支对应的分支，使用`git checkout -b branch-name origin/branch-name`，本地和远程分支的名称最好一致；
    - 建立本地分支和远程分支的关联，使用`git branch --set-upstream branch-name origin/branch-name`；
- `git rebase`: 把本地未push的分叉提交历史整理成直线,然后用`git log`查看

## 2.6 标签管理

- `git tag v1.0`:给当前分支打上标签v1.0
- `git tag v0.9 f52c643`: 指定给某一个历史提交打上标签。标签总是和某个commit挂钩，若该commit出现在不同的2个分支上，这2个分支上都可以看到标签。
- `git tag`:查看标签
- `git tag -a <tagname> -m "blablabla..."`可以指定标签信息；
   - -a: 标签名字
   - -m:说明文字 
- `git show <tagname>`查看标签信息
- `git tag -d v0.1`:删除本地标签
- `git push origin :refs/tags/<tagname>`可以删除一个远程标签
- `git push origin <tagname>`可以推送一个本地标签；
- `git push origin --tags`可以推送全部未推送过的本地标签；

## 2.7、子模块

- 帮助文档:`git help submodule`

- CLone包含子模块的项目：

  在clone后需要使用`git submodule update --init --recursive`来初始化子模块内容。

- 添加/删除子模块：

  ```shell
  添加：
  $ git submodule add <url> <repo_name>
  例子：
  $ git submodule add https://github.com/iphysresearch/GWToolkit.git GWToolkit
  
  删除：
  git rm --cached GWToolkit
  rm -rf GWToolkit
  并手动删除下面下个文件中相关的内容
  ```

  添加后会多出如下文件

  1. `.gitmodules文件`：用于保存子模块的信息。
  2. `.git/config`: 中会多出一块关于子模块的信息内容
  3. `.git/modules/GwToolkit`: 子模块文件

- 更新子模块：

  - 更新项目内子模块到最新版本：`git submodule update`
  - 更新子模块为远程项目的最新版本：`git submodule update --remote`

