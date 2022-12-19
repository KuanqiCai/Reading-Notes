# 一、学习资源

- [学习web开发](https://developer.mozilla.org/zh-CN/docs/Learn)

- JavaScript
  - [C语言中文网教程](http://c.biancheng.net/view/5355.html)

# 二、Web概述

## 1. 本地测试服务器

### 1.1 本地文件和远程文件

通常HTML不论是本地文件还是远程文件，都可以直接在浏览器中打开，打开时：

- 本地打开的HTML文件：地址栏开头是`file://`
- 远程打开的HTML文件：地址栏开头是`http://`

但如下情况，HTML无法被本地打开：

- **它们具有异步请求。** 如果你只是从本地文件运行示例，一些浏览器（包括 Chrome）将不会运行异步请求（请参阅 [从服务器获取数据](https://developer.mozilla.org/zh-CN/docs/Learn/JavaScript/Client-side_web_APIs/Fetching_data)）。这是因为安全限制（更多关于 Web 安全的信息，请参阅 [站点安全](https://developer.mozilla.org/zh-CN/docs/Learn/Server-side/First_steps/Website_security)）。
- **它们具有服务端代码。** 服务器端语言（如 PHP 或 Python）需要一个特殊的服务器来解释代码并提供结果。

为此，我们需要一个本地HTTP服务器

### 1.2 运行一个简单的本地HTTP服务器

1. 在项目目录下：

   ```shell
   # 参数7800如果不加，默认端口是8000
   python -m http.server 7800
   ```

2. 在浏览器地址栏中打开:

   ```
   localhost:8000
   ```

   

## 2. 设计网站

### 2.1 计划：

考虑如下问题：

1. **网站的主题是什么？** 你喜欢狗、上海、还是吃豆人？
2. **基于所选主题要展示哪些信息？** 写下标题和几段文字，构思一个用于展示的图像。
3. **网站采用怎样的外观？** 用高阶术语来说，背景颜色是什么？使用哪种字体比较合适：正式的、卡通的、粗体、瘦体？

### 2.2 主题颜色

可以使用使用[色彩选择器](https://developer.mozilla.org/zh-CN/docs/Web/CSS/CSS_Colors/Color_picker_tool)挑选心仪的颜色，得到十六进制码，如：`#660066`

### 2.3 图片

使用[Google Images](https://www.google.com/imghp?gws_rd=ssl)找需要的图片。

为了避免版权问题：

1. 搜索栏下有个tools，然后选择usage rights
2. 然后选择Creative Commons licenses

### 2.4 字体

使用[Google Fonts](https://fonts.google.com/)来寻找想要的字体

For more details about using Google Fonts, see [this page](https://developers.google.com/fonts/docs/getting_started)

## 3. 处理文件

一个网站包含许多文件：文本内容、代码、样式表、媒体内容，等等。

- 本地网站相关的文件应该保存在一个单独的文件夹`web-projects`中
  - 在这个文件中，每一个子文件夹代表一个网站`first-site`
- 文件命名时：
  - 尽量全都使用小写
  - 不使用空格和下划线，而用连字符`-`来分割单词

- 网站结构：

  - **`index.html`**：这个文件一般会包含主页内容，也就是人们第一次进入网站时看到的文字和图片。使用文本编辑器，创建一个名为`index.html`的新文件，并将其保存在`test-site`文件夹内。
  - **`images` 文件夹**：这个文件夹包含网站上使用的所有图片。在 `test-site` 文件夹内创建一个名为 `images` 的文件夹。
  - **`styles` 文件夹**：这个文件夹包含用于设置内容样式的 CSS 代码（例如，设置文本和背景颜色）。在你的 `test-site` 文件夹内创建一个名为 `styles` 的文件夹。
  - **`scripts` 文件夹**：这个文件夹包含所有用于向网站添加交互功能的 JavaScript 代码（例如，点击时加载数据的按钮）。在 `test-site` 文件夹内创建一个名为 `scripts` 的文件夹。、

- 文件路径

  一个例子：
  
  ```html
  <!DOCTYPE html>
  <html>
    <head>
      <meta charset="utf-8">
      <title>My test page</title>
    </head>
    <body>
      <img src="images/test.jpg" alt="My test image">
    </body>
  </html>
  ```
  
  文件路径的一些通用规则：
  
  - 若引用的目标文件与 HTML 文件同级，只需直接使用文件名，例如：`my-image.jpg`。
  - 要引用子目录中的文件，请在路径前面写上目录名，再加上一个正斜杠。例如：`subdirectory/my-image.jpg`。
  - 若引用的目标文件位于 HTML 文件的**上级**，需要加上两个点。举个例子，如果 `index.html` 在 `test-site` 的一个子文件夹内，而 `my-image.jpg` 在 `test-site` 内，你可以使用`../my-image.jpg` 从 `index.html` 引用 `my-image.jpg`。
  - 以上方法可以随意组合，比如：`../subdirectory/another-subdirectory/my-image.jpg`。

## 4. HTML

参见笔记[[HTML]]

## 5. CSS

参见笔记[[CSS]]

## 6. JavaScript

参见笔记[[JavaScript]]

## 7. 发布网站

有各种各样的发布网站的方式：

### 7.1 购买域名

- 主机服务 — 在主机服务提供商的 [Web 服务器](https://developer.mozilla.org/zh-CN/docs/Learn/What_is_a_web_server)上租用文件空间。将你网站的文件上传到这里，然后服务器会提供 Web 用户需求的内容。
- [域名](https://developer.mozilla.org/zh-CN/docs/Learn/Understanding_domain_names)——一个可以让人们访问的独一无二的地址，比如 `http://www.mozilla.org`，或 `http://www.bbc.co.uk` 。你可以从**域名注册商**租借域名。
- 此外需要一个 [文件传输协议](https://developer.mozilla.org/zh-CN/docs/Glossary/FTP) 程序 ( 点击[钻研在网络上做某些事情要花费多少：软件](https://developer.mozilla.org/zh-CN/docs/Learn/Common_questions/How_much_does_it_cost#软件)查看详细信息 ) 来将网站文件上传到服务器。
- 一些免费服务： [Neocities](https://neocities.org/) ， [Blogspot](https://www.blogger.com/) ，和 [Wordpress](https://wordpress.com/)

### 7.2 使用在线工具

- GitHub
- Google App Engine 是一个让你可以在 Google 的基础架构上构建和运行应用的强劲平台——无论你是需要从头开始构建多级 web 应用还是托管一个静态网站。参阅[How do you host your website on Google App Engine?](https://developer.mozilla.org/zh-CN/docs/Learn/Common_questions/How_do_you_host_your_website_on_Google_App_Engine)以获取更多信息。

#### 7.2.1 使用GitHub托管网站

1. 创建一个库，库名格式`username.github.io`

   - 比如`fernweh-yang.github.io`

   - 勾选Add a README file

2. 将网站的文件都上传进仓库，保证文件夹里有一个index.html

3. 在1/2分钟的部署后，就可以用`fernweh-yang.github.io`来访问网站了

### 7.3 使用Web集成开发环境

有许多 web 应用能够仿真一个网站开发环境。你可以在这种应用——通常只有一个标签页——里输入 HTML、CSS 和 JavaScript 代码然后像显示网页一样显示代码的结果。通常这些工具都很简单，对学习很有帮助，而且至少有免费的基本功能，它们在一个独特的网址显示你提交的网页。不过，这些应用的基础功能很有限，而且应用通常不提供空间来存储图像等内容。

- [JSFiddle](https://jsfiddle.net/)
- [Glitch](https://glitch.com/)
- [JSBin](http://jsbin.com/)
- [CodePen](https://codepen.io/)

## 8. web如何工作

### 8.1 基本概念

- **客户端和服务器**: 连接到互联网的计算机被称作客户端和服务器。

  - 客户端是典型的 Web 用户入网设备（比如，你连接了 Wi-Fi 的电脑，或接入移动网络的手机）和设备上可联网的软件（通常使用像 Firefox 和 Chrome 的浏览器）。
  - 服务器是存储网页，站点和应用的计算机。当一个客户端设备想要获取一个网页时，一份网页的拷贝将从服务器上下载到客户端机器上来在用户浏览器上显示。

- **网络连接**: 允许你在互联网上发送和接受数据。基本上和你家到商店的街道差不多。

- **TCP/IP**: 传输控制协议和因特网互连协议是定义数据如何传输的通信协议。这就像你去商店购物所使用的交通方式，比如汽车或自行车（或是你能想到的其他可能）。

- **DNS**: 域名系统服务器像是一本网站通讯录。当你在浏览器内输入一个网址时，浏览器获取网页之前将会查看域名系统。浏览器需要找到存放你想要的网页的服务器，才能发送 HTTP 请求到正确的地方。就像你要知道商店的地址才能到达那。

- **HTTP**: 超文本传输协议是一个定义客户端和服务器间交流的语言的协议（[protocol](https://developer.mozilla.org/zh-CN/docs/Glossary/Protocol) ）。就像你下订单时所说的话一样。

- **组成文件**: 一个网页由许多文件组成，就像商店里不同的商品一样。这些文件有两种类型：
  - **代码** : 网页大体由 HTML、CSS、JavaScript 组成，不过你会在后面看到不同的技术。
  - **资源** : 这是其他组成网页的东西的集合，比如图像、音乐、视频、Word 文档、PDF 文件。

### 8.2 到底发生了什么：

当你在浏览器里输入一个网址时（在我们的例子里就是走向商店的路上时）：

1. 浏览器在域名系统（DNS）服务器上找出存放网页的服务器的实际地址（找出商店的位置）。
2. 浏览器发送 HTTP 请求信息到服务器来请拷贝一份网页到客户端（你走到商店并下订单）。这条消息，包括其他所有在客户端和服务器之间传递的数据都是通过互联网使用 TCP/IP 协议传输的。
3. 服务器同意客户端的请求后，会返回一个“200 OK”信息，意味着“你可以查看这个网页，给你～”，然后开始将网页的文件以数据包的形式传输到浏览器（商店给你商品，你将商品带回家）。
4. 浏览器将数据包聚集成完整的网页然后将网页呈现给你（商品到了你的门口 —— 新东西，好棒！）。

### 8.3 解析组成文件的顺序

当浏览器向服务器发送请求获取 HTML 文件时，HTML 文件通常包含`<link>`和`<script>`元素，这些元素分别指向了外部的 [CSS](https://developer.mozilla.org/zh-CN/docs/Learn/CSS) 样式表文件和 [JavaScript](https://developer.mozilla.org/zh-CN/docs/Learn/JavaScript) 脚本文件。了解这些文件被[浏览器解析](https://developer.mozilla.org/zh-CN/docs/Web/Performance/How_browsers_work#解析)的顺序是很重要的：

- 浏览器首先解析 HTML 文件，并从中识别出所有的 `<link>` 和 `<script>` 元素，获取它们指向的外部文件的链接。
- 继续解析 HTML 文件的同时，浏览器根据外部文件的链接向服务器发送请求，获取并解析 CSS 文件和 JavaScript 脚本文件。
- 接着浏览器会给解析后的 HTML 文件生成一个 [DOM](https://developer.mozilla.org/zh-CN/docs/Web/API/Document_Object_Model) 树（在内存中），会给解析后的 CSS 文件生成一个 [CSSOM](https://developer.mozilla.org/zh-CN/docs/Glossary/CSSOM) 树（在内存中），并且会[编译和执行](https://developer.mozilla.org/zh-CN/docs/Web/Performance/How_browsers_work#其他过程)解析后的 JavaScript 脚本文件。
- 伴随着构建 DOM 树、应用 CSSOM 树的样式、以及执行 JavaScript 脚本文件，浏览器会在屏幕上绘制出网页的界面；用户看到网页界面也就可以跟网页进行交互了。

### 8.4 DNS解析

真正的网址看上去并不像你输入到地址框中的那样美好且容易记忆。它们是一串数字，像 `63.245.217.105`。

这叫做 [IP 地址](https://developer.mozilla.org/zh-CN/docs/Glossary/IP_Address)，它代表了一个互联网上独特的位置。然而，它并不容易记忆，不是吗？那就是域名系统（DNS）被发明的原因。它们是将你输入浏览器的地址（像 "mozilla.org"）与实际 IP 地址相匹配的特殊的服务器。

网页可以通过 IP 地址直接访问。您可以通过在 [DNS 查询工具](https://www.nslookup.io/website-to-ip-lookup/) 等工具中输入域名来查找网站的 IP 地址。

### 8.5 数据包详解

前面我们用“包”来描述了数据从服务器到客户端传输的格式。这是什么意思？基本上，当数据在 Web 上传输时，是以成千上万的小数据块的形式传输的。大量不同的用户都可以同时下载同一个网页。如果网页以单个大的数据块形式传输，一次就只有一个用户下载，无疑会让 Web 非常没有效率并且失去很多乐趣。

### 8.6 扩展阅读

- [互联网是如何工作的](https://developer.mozilla.org/zh-CN/docs/Learn/Common_questions/How_does_the_Internet_work)
- [HTTP — 一种应用级协议](https://dev.opera.com/articles/http-basic-introduction/)
- [HTTP：让我们开始吧！](https://dev.opera.com/articles/http-lets-get-it-on/)
- [HTTP：响应代码](https://dev.opera.com/articles/http-response-codes/)
