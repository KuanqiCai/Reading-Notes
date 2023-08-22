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
        this->age = age;
        this->score = score;
    }

    void setNewAgeScore2(int age, float score) {
        this->age = age;
        this->score = score;
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

结果：
小明的年龄是15，成绩是92.5

小明的年龄是16，成绩是98.66

小明的年龄是17，成绩是96.75


2. This 是什么
this 实际上是成员函数的一个形参，在调用成员函数时将对象的地址作为实参传递给 this。不过 this 这个形参是隐式的，它并不出现在代码中，而是在编译阶段由编译器默默地将它添加到参数列表中，本质上，this 作为隐式形参，是成员函数的局部变量，所以只能用在成员函数的内部，并且只有在通过对象调用成员函数时才给 this 赋值。

** Class定义的函数中用到 Class定义的变量的时候需要用：“ this-> ” 来修改class中的变量。**


## 构造函数

1. 是什么？

构造函数是类的一种特殊的成员函数，它会在每次创建类的新对象时执行

怎样定义？

与类名同名，没有返回值，可以被重载

有什么用？

通常用来做初始化工作

2. 定义构造函数

```c++
#include <iostream>

using namespace std;

class student {
public :
    char *name;
    int age;

    // 构造函数
    student() {
        cout << "执行无参构造函数" << endl;
    }

    student(char *name) {
        cout << "执行含有一个参数的构造函数" << endl;
    }

    student(char *name, int age) {
        cout << "执行含有两个参数的构造函数" << endl;
    }

};
```
3. 调用构造函数

调用方式很简单，只要在创建对象时，会自动调用

- ()：如果创建对象不需要传递实参，可以不写()

```c++
int main() {
    student stu1;
    student stu2("张三");
    student stu3("张三", 28);

    return 0;
}
```
结果：

执行无参构造函数

执行含有一个参数的构造函数

执行含有两个参数的构造函数


## 析构函数

1. 是什么？

一种在对象销毁时，自动调用的函数

怎样定义？

析构函数名称与类名称相同，只是在前面加了个波浪号~作为前缀，它不会返回任何值，也不能带有任何参数，不能被重载

有什么用？

一般用于释放资源，例如关闭打开的文件，打开的网络socket套接字等

2. 定义析构函数

```c++
#include <iostream>
#include <string>

using namespace std;

class Student {
public :
    char *name;
    int age;

    // 构造函数
    Student() {
        cout << "执行无参构造函数" << endl;
    }

    Student(char *name) {
        cout << "执行含有一个参数的参构造函数" << endl;
    }

    Student(char *name, int age) {
        cout << "执行含有两个参数的构造函数" << endl;
    }

    // 析构函数
    ~Student() {
        cout << "执行析构函数" << endl;
    }
};
```

3. 调用析构函数

不需要调用，在对象销毁时，自动调用

```c++
int main() {
    // 创建对象时，自动调用构造函数
    Student *pStu1 = new Student;
    Student *pStu2 = new Student;
    Student *pStu3 = new Student;

    // 销毁对象时，自动调用析构函数
    delete pStu1;
    delete pStu2;
    delete pStu3;

    return 0;
}
```
结果：

执行无参构造函数

执行无参构造函数

执行无参构造函数

执行析构函数

执行析构函数

执行析构函数


### 成员访问限定符：private、protected、public

1. 是什么？

一句话：一种权限标记，就是你能不能使用该类中的成员

2. 为什么要用？

一个对象，由很多数据（成员变量）+很多函数（成员函数）组成，有些时候定义的成员变量数据比较重要不想通过 (对象.成员变量=xxx) 或 (对象指针->成员变量=xxx) 的方式直接修改，此时我们就需要使用到权限的限定就好比我们都是一家人，都可以用公用的数据，对外人是不能用的。

3. 3种方式

通过 public、protected、private 三个关键字来控制成员变量和成员函数的访问权限，被称为成员访问限定符

- public：公有的

- protected：受保护的

- private：私有的

在类的内部（定义类的代码内部），无论成员被声明为 public、protected 还是 private，都是可以互相访问的，没有访问权限的限制

在类的外部（定义类的代码之外），只能通过对象或指针访问public修饰的成员，不能访问 private、protected 修饰的成员

4. 示例
```c++
#include <iostream>

using namespace std;


class Student {
private:
    char *name;
    int age;
    float score;

public:
    // 成员函数的定义
    void set_name(char *name) {
        this->name = name;
    }

    void set_age(int age) {
        this->age = age;
    }

    void set_score(float score) {
        this->score = score;
    }

    void show() {
        cout << this->name << "的年龄是" << this->age << "，成绩是" << this->score << endl;
    }
};


int main() {
    // 在栈上创建对象
    Student stu;
    stu.set_name("张三");
    stu.set_age(15);
    stu.set_score(92.5f);
//    cout << stu.name << "\n";  // 失败，因为name是私有的，不能在类的外部通过对象访问
    stu.show();

    // 在堆上创建对象
    Student *pstu = new Student;
    pstu->set_name("李四");
    pstu->set_age(16);
    pstu->set_score(96);
//    cout << pstu->name << "\n";  // 失败，因为name是私有的，不能在类的外部通过对象访问
    pstu->show();

    return 0;
}
```

运行结果：

张三的年龄是15，成绩是92.5

李四的年龄是16，成绩是96

注意

- private 后面的成员都是私有的，直到有 public 出现才会变成共有的

- public 之后再无其他限定符，所以 public 后面的成员都是共有的

- private 的成员和 public 的成员的次序任意，既可以先出现 private 部分，也可以先出现 public 部分

- 如果既不写 private 也不写 public，就默认为 private

- 在一个类体中，private 和 public 可以分别出现多次

- 每个部分的有效范围到出现另一个访问限定符或类体结束时（最后一个右花括号）为止。但是为了使程序清晰，应该养成这样的习惯，使每一种成员访问限定符在类定义体中只出现一次

例：
```c++
class Student{
private:
    char *m_name;
private:
    int m_age;
    float m_score;
public:
    void setname(char *name);
    void setage(int age);
public:
    void setscore(float score);
    void show();
};
```
### 封装

1. 成员变量

成员变量，说到底就是对象的一些可以用来存储数据的变量而已，如果不希望通过外部使用这些变量，就private修饰

2. 成员函数

成员函数，说到底就是对象的一些可以调用的函数而已

- 如果希望通过外部调用这个函数，就用public修饰
- 
- 如果不希望外部调用这个函数，那么就private修饰
- 
- 还有一种情况，如果定义了private成员变量，那么可以定义对应的public成员函数，这样做的好处

这个私有的成员变量，不能直接通过外部对象访问，但是可以通过这个公有的函数间接访问
一般给某个私有的成员变量，创建对应的2个公有成员函数

给成员变量赋值的函数通常称为 set 函数，它们的名字通常以set开头，后跟成员变量的名字
读取成员变量的值的函数通常称为 get 函数，它们的名字通常以get开头，后跟成员变量的名字

3. 总结

这种将成员变量声明为 private、将部分成员函数声明为 public 的做法体现了类的封装性

** 所谓封装，是指尽量隐藏类的内部实现，只向用户提供有用的成员函数 **

你可能会说，额外添加 set 函数和 get 函数多麻烦，直接将成员变量设置为 public 多省事！确实，这样做 99.9% 的情况下都不是一种错误，我也不认为这样做有什么不妥；但是，将成员变量设置为 private 是一种软件设计规范，尤其是在大中型项目中，还是请大家尽量遵守这一原则

