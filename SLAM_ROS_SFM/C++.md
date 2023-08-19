# C++ 快速理解
## 对象

1. 联想C语言结构体

   C++是一门面向对象的编程语言，理解 C++，首先要理解**类（Class**和**对象（Object**这两个概念 C++ 中的类（Class）可以看做C语言中结构体（Struct）的升级版, 结构体是一种构造类型，可以包含若干成员变量，每个成员变量的类型可以不同。

通过结构体来定义结构体变量，每个变量拥有相同的性质

1.1 C++ 代码

C++ 中的类也是一种构造类型，但是进行了一些扩展，类的成员不但可以是变量，还可以是函数

通过类定义出来的变量也有特定的称呼，叫做“对象”

   ```C++
   #include <stdio.h>

// 通过class关键字类定义类
class Student {
public:
    // 下方的变量目的：规定数据类型
    // 类包含的变量
    char *name;
    int age;
    float score;

    // 下方函数目的：规定对数据的处理方式
    // 类包含的函数
    void say() {
        printf("%s的年龄是 %d，成绩是 %f\n", name, age, score);
    }
};

int main() {
    // 通过类来定义变量，即创建对象
    class Student boy;  // 也可以省略关键字class

    // 为类的成员变量赋值
    boy.name = "小明";
    boy.age = 17;
    boy.score = 90.3f;

    // 调用类的成员函数
    boy.say();

    return 0;
}

   ```

运行结果： 小明的年龄是17,成绩是90.300003

2. 说明

   class 和 public 都是 C++ 中的关键字，初学者请先忽略 public（后续会深入讲解），把注意力集中在 class

    ～ C语言中的 struct 只能包含变量

    ～ C++ 中的 class 除了可以包含变量，还可以包含函数

   display() 是用来处理成员变量的函数，在C语言中，我们将它放在了 struct Student 外面，它和成员变量是分离的而在 C++ 中，将它放在了 class Student 内部，使它和成员变量聚集在一起，看起来更像一个整体结构体和类都可以看做一种由用户自己定义的复杂数据类型

在C语言中可以通过结构体名来定义变量，在 C++ 中可以通过类名来定义变量，不同的是

    ～ 通过结构体定义出来的变量还是叫变量
    
    ～ 通过类定义出来的变量有了新的名称，叫做对象（Object）
    
有些资料也将类的成员变量称为属性（Property），将类的成员函数称为方法（Method）

在 C++ 中，多了一层封装，就是类（Class）。类由一组相关联的函数、变量组成，你可以将一个类或多个类放在一个源文件，使用时引入对应的类就可以

![](https://github.com/KuanqiCai/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/ROS/c%2B%2B.png)

## 对象

1. 类

类，就是一个模板，可以规定有什么数据，怎样处理数据
我们把类中定义的数据（就是变量），称之为**成员变量**，有的人也叫做属性
我们把类中定义的函数，称之为**成员函数**，有的人也叫做方法

2. 对象
   
对象，就是用类创建的一个空间，这个空间中可以使用成员变量，可以使用成员函数

3. 定义类

4. ```C++
   class Student{
   public:
    // 成员变量
    char *name;
    int age;
    float score;

    // 成员函数
    void say(){
        cout<<name<<"的年龄是"<<age<<"，成绩是"<<score<<endl;
    }
};
```
Tip:

- 上面的代码创建了一个 Student 类，它包含了 3 个成员变量和 1 个成员函数

- class是 C++ 中新增的关键字，专门用来定义类

- Student是类的名称；类名的首字母一般大写（就是大驼峰命名法），以和其他的标识符区分开

- { }内部是类所包含的成员变量和成员函数，它们统称为类的成员（Member）

- public也是 C++ 的新增关键字，它只能用在类的定义中，表示类的成员变量或成员函数具有“公开”的访问权限

注意：

- 在类定义的最后有一个分号;它是类定义的一部分，表示类定义结束了，不能省略

类只是一个模板（Template），编译后不占用内存空间，所以在定义类时不能对成员变量进行初始化，因为没有地方存储数据。只有在创建对象以后才会给成员变量分配内存，这个时候就可以赋值了

5. 创建对象

有了 Student 类后，就可以通过它来创建对象了，例如：

‵‵‵C++
Student liLei;  // 创建对象
```

Student是类名，liLei是对象名。这和使用基本类型定义变量的形式类似：

```c++
int num;  // 定义整型变量
```

从这个角度考虑，我们可以把 Student 看做一种新的数据类型，把 liLei 看做一个变量
在创建对象时，class 关键字可要可不要，但是出于习惯我们通常会省略掉 class 关键字，例如：

```c++
class Student LiLei;  // 正确
Student LiLei;  // 同样正确
```

除了创建单个对象，还可以创建对象数组：

```c++
Student stus[100];
```
该语句创建了一个 stus 数组，它拥有100个元素，每个元素都是 Student 类型的对象


## 访问成员

1. 使用.访问成员
   创建对象以后，可以使用点号.来访问成员变量和成员函数，这和通过结构体变量来访问它的成员类似如下所示：

code:

```c++
#include <iostream>

using namespace std;

// 类通常定义在函数外面
class Student {
public:
    // 成员变量
    char *name;
    int age;
    float score;

    // 成员函数
    void say() {
        cout << name << "的年龄是" << age << "，成绩是" << score << endl;
    }
};

int main() {
    // 创建对象
    Student boy;
    // 通过.可以使用对象中的成员
    boy.name = "小明";
    boy.age = 15;
    boy.score = 92.5f;
    boy.say();

    // 创建对象2
    Student girl;
    // 通过.可以使用对象2中的成员
    girl.name = "菲菲";
    girl.age = 18;
    girl.score = 88.7f;
    girl.say();


    return 0;
}

```

结果：
小明的年龄是15，成绩是92.5；
菲菲的年龄是18，成绩是88.7；

解析
![](https://github.com/KuanqiCai/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/ROS/c_plus_plus_cite.png)


2. 使用->访问成员
上面代码中创建的对象 boy在栈上分配内存，需要使用&获取它的地址，例如：

```c++
Student boy;
Student *pStu = &boy;
```

pStu 是一个指针，它指向 Student 类型的数据，也就是通过 Student 创建出来的对象
当然，你也可以在堆上创建对象，这个时候就需要使用前面讲到的new关键字，例如：

```c++
Student *pStu = new Student;
```

通过 new 创建出来的对象，它在堆上分配内存，没有名字，必须使用一个指针变量来接收这个指针，否则以后再也无法找到这个对象了，更没有办法使用它

区别？
-栈内存是程序自动管理的
-堆内存由程序员管理，对象使用完毕后需要用 delete 删除
有了对象指针后，可以通过箭头->来访问对象的成员变量和成员函数，这和通过结构体指针来访问它的成员类似

示例：

```c++
#include <iostream>

using namespace std;

class Student {
public:
    char *name;
    int age;
    float score;

    void say() {
        cout << name << "的年龄是" << age << "，成绩是" << score << endl;
    }
};

int main() {
    // 创建1个对象，让pStu这个指针变量指向它
    Student *pStu = new Student;
    // 既然pStu指向了一个对象（内存空间），那么就可以调用这个内存空间中的某个成员
    pStu->name = "小明";
    pStu->age = 15;
    pStu->score = 92.5f;
    // 当然也可以调用它里面的函数
    pStu->say();

    // 当不要pStu指向的对象时，需要用delete进行删除（就是释放那个内存空间）
    delete pStu;

    return 0;
}
```

注意：
对不再使用的对象，记得用 delete 进行回收空间，这是一种良好的编程习惯

## this 指针

1. 引入

```c++
#include <iostream>

using namespace std;

class Student {
public:
    char *name;
    int age;
    float score;

    void say() {
        cout << name << "的年龄是" << age << "，成绩是" << score << endl;
    }

    void setNewAgeScore(int ageTemp, float scoreTemp) {
        age = ageTemp;
        score = scoreTemp;
    }

    void setNewAgeScore2(int age, float score) {
        age = age;
        score = score;
    }
};

int main() {

    // 创建一个对象
    Student *pStu = new Student;

    // 调用对象的成员变量
    pStu->name = "小明";
    pStu->age = 15;
    pStu->score = 92.5f;

    // 调用对象的成员函数
    pStu->say();

    // 设置新的年龄、分数
    pStu->setNewAgeScore(16, 98.66f);
    pStu->say();
    pStu->setNewAgeScore2(17, 96.75);
    pStu->say();

    delete pStu;  //删除对象

    return 0;
}
```
