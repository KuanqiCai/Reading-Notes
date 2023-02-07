# 1.如何打包python项目

- 资料总结：
  - https://www.osgeo.cn/python-packaging/tutorials/packaging-projects.html

- 下面以打包一个example_pkg项目为例子：

## 1.1 创建项目文件

```python
├─packaging
│  └─example_pkg
│          __init__.py
```

其中`__init__.py`加入如下代码

```python
# 用于验证是否正确安装
name = "example_pkg"
```

下面发布example_pkg这个包

## 1.2 创建包文件

```shell
├─packaging
│  │ LICENSE
│  │ README.md
│  │ setup.py
│  │
│  └─example_pkg
│          __init__.py
```



### 1.2.1 setup.py

`setup.py` 是的生成脚本 [setuptools](https://www.osgeo.cn/python-packaging/key_projects.html#setuptools) . 它告诉安装工具关于您的包（如名称和版本）以及要包括哪些代码文件。

```python
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="example-pkg-yang",
    version="0.0.1",
    author="Yang Xu",
    author_email="xudashuai512@gmail.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
```

- `name` 是包的名字。只能包含字母、数字和 `_` 和 `-`。必须是唯一的，不能和pypi.org上已有的包同名。
- `version` 包版本见 [PEP 440](https://www.python.org/dev/peps/pep-0440) 有关版本的详细信息。
- `author` 和 `author_email` 用于标识包的作者。
- `description` 是包的一个简短的一句话摘要。
- `long_description` 是对包的详细描述。这在python包索引的包细节包中显示。通常长描述从 `README.md`中读取。
- `long_description_content_type` 告诉索引用于长描述的标记类型。这里用的markdown。
- `url` 是项目主页的URL。对于许多项目，这只是到GitHub、Gitlab、BitBucket或类似的代码托管服务的链接。
- `packages` 是所有python的列表 [import packages](https://www.osgeo.cn/python-packaging/glossary.html#term-import-package) 应该包括在 [distribution package](https://www.osgeo.cn/python-packaging/glossary.html#term-distribution-package) . 我们可以使用 `find_packages()` 自动发现所有包和子包。这里example_pkg是唯一的包裹。
- `classifiers` 给出索引和 [pip](https://www.osgeo.cn/python-packaging/key_projects.html#pip) 关于您的包的一些附加元数据。这里包只与python 3兼容，在MIT许可下获得许可，并且是独立于操作系统的。您应该始终至少包括您的包所使用的Python的哪个版本、包所使用的许可证以及包将使用的操作系统。有关分类器的完整列表，请参阅https://pypi.org/classifiers/。

### 1.2.2 Readme.md

1.2.1中提到的long_description:

```markdown
# Example Package
This is a simple example package.
```

### 1.2.3 LICENSE

对于上传到python包索引的每个包来说，包含一个许可证是很重要的。这会告诉安装您的软件包的用户可以使用您的软件包的条款。有关选择许可证的帮助，请参阅https://choosealelicense.com/。选择许可证后，打开 `LICENSE` 并输入许可证文本。例如，如果您选择了MIT许可证：

```
Copyright (c) 2018 The Python Packaging Authority

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```



## 1.3 得到分配包

为了得到包裹的[distribution packages](https://www.osgeo.cn/python-packaging/glossary.html#term-distribution-package) 

1. 需要setuptools 和 wheel

   ```
   python3 -m pip install --user --upgrade setuptools wheel
   ```

2. 在setup.py所在的文件夹下执行```python3 setup.py sdist bdist_wheel```

   得到文件夹dist：

   ```shell
   C:\Users\51212\Desktop\testeverything\packaging> tree /f
   │  LICENSE
   │  README.md
   │  setup.py
   │
   ├─build
   │  ├─bdist.win-amd64
   │  └─lib
   │      └─example_pkg
   │              __init__.py
   │
   ├─dist
   │      example-pkg-yang-test-0.0.1.tar.gz
   │      example_pkg_yang_test-0.0.1-py3-none-any.whl
   │
   ├─example_pkg
   │      __init__.py
   │
   └─example_pkg_yang_test.egg-info
           dependency_links.txt
           PKG-INFO
           SOURCES.txt
           top_level.txt
   ```

   

   

## 1.4 发布包

因为这里并不把这个测试包真的发出到[Python Package Index (PyPI)](https://www.osgeo.cn/python-packaging/glossary.html#term-python-package-index-pypi)，而是发出到[test.pypi.org](https://test.pypi.org/)。

1. TestPyPI可以用于测试发布我们的代码，而不需要担心干扰真实的版本号。但需要[创建一个账号](https://test.pypi.org/account/register/)

2. 安装 [twine](https://www.osgeo.cn/python-packaging/key_projects.html#twine) 用来上传分发包

   ```
   python3 -m pip install --user --upgrade twine
   ```

3. 上传包

   ```
   python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
   ```

   得到：

   ```shell
   PS C:\Users\51212\Desktop\testeverything\packaging> python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
   Uploading distributions to https://test.pypi.org/legacy/
   Enter your username: tianxiadiyishuai
   Enter your password:
   Uploading example_pkg_yang_test-0.0.1-py3-none-any.whl
   100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.5/5.5 kB • 00:00 • ?
   Uploading example-pkg-yang-test-0.0.1.tar.gz
   100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.1/5.1 kB • 00:00 • ?
   
   View at:
   https://test.pypi.org/project/example-pkg-yang-test/0.0.1/
   ```

## 1.5 下载包

- 从TestPyPi上下载包

  可在1.4的https://test.pypi.org/project/example-pkg-yang-test/0.0.1/查看下载指令

  ```
  pip install -i https://test.pypi.org/simple/ example-pkg-yang-test==0.0.1
  ```

- 检查一下：

  ```
  (.env) PS C:\Users\51212\Desktop\testeverything\packaging> pip list
  Package               Version Editable project location
  --------------------- ------- --------------------------------------------------
  cloudpickle           2.2.0
  example-pkg-yang-test 0.0.1
  gym                   0.26.0
  gym-examples          0.0.1   c:\users\51212\desktop\testeverything\gym-examples
  gym-notices           0.0.8
  numpy                 1.23.5
  pip                   22.3.1
  pygame                2.1.0
  setuptools            65.5.0
  ```

- 使用一下：

  ```python
  >>> import example_pkg		# 1.1中包的名字叫example_pkg
  >>> example_pkg.name		# 1.1中__init__.py中的代码
  'example_pkg'
  ```

## 1.6 本地卸载

```shell
pip uninstall example-pkg-yang-test	# 1.2.1中定义的包名
```

## !!!如果不需要发布

可以直接创建setup后安装

1. 文件夹格式如下：

   ```
   (.env) PS C:\Users\51212\Desktop\testeverything\testpkg> tree /f
   │  README.md
   │  setup.py
   │
   ├─example_pkg
   │      __init__.py
   ```

2. 安装这个包到环境中去

   ```
   (.env) PS C:\Users\51212\Desktop\testeverything\testpkg> pip install -e .
   ```

   



# 2. 数据操作

## 2.1 判断数据类型

使用`if not isinstance`来判断：

```python
# 判断是否是浮点型
a=2
if not isinstance(a,float):
	raise TypeError("wrong")
    
# 判断是否是tuple
b=[1,2,3]
if not isinstance(b,tuple):
    raise TypeError("not tuple")
```

# 3. 类相关
## 3.1 修饰符：classmethod
classmethod 修饰符对应的函数不需要实例化，不需要 self 参数，但第一个参数需要是表示自身类的 cls 参数，可以来调用类的属性，类的方法，实例化对象等.  
例子：
```
class A(object):

    # 属性默认为类属性（可以给直接被类本身调用）
    num = "类属性"

    # 实例化方法（必须实例化类之后才能被调用）
    def func1(self): # self : 表示实例化类后的地址id
        print("func1")
        print(self)

    # 类方法（不需要实例化类就可以被类本身调用）
    @classmethod
    def func2(cls):  # cls : 表示没用被实例化的类本身
        print("func2")
        print(cls)
        print(cls.num)
        cls().func1()

    # 不传递传递默认self参数的方法（该方法也是可以直接被类调用的，但是这样做不标准）
    def func3():
        print("func3")
        print(A.num) # 属性是可以直接用类本身调用的
    
# A.func1() 这样调用是会报错：因为func1()调用时需要默认传递实例化类后的地址id参数，如果不实例化类是无法调用的
A.func2()
A.func3()
```

