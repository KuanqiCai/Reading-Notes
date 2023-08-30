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

### 继承和派生

1. 是什么？

C++ 中的继承是类与类之间的关系，与现实世界中的继承类似

例如：儿子继承父亲的财产

继承（Inheritance）可以理解为一个类从另一个类获取成员变量和成员函数的过程

例如：

类B继承于类A，那么B就拥有A的成员变量和成员函数

- 在C++中，派生（Derive）和继承是一个概念，只是站的角度不同，继承是儿子接收父亲的产业，派生是父亲把产业传承给儿子

被继承的类称为父类或基类，继承的类称为子类或派生类

“子类”和“父类”通常放在一起称呼，“基类”和“派生类”通常放在一起称呼

2. 为什么？

子类除了拥有父类的成员，还可以定义自己的新成员，以增强类的功能

以下是两种典型的使用继承的场景：

- 当你创建的新类与现有的类相似，只是多出若干成员变量或成员函数时，可以使用继承，这样不但会减少代码量，而且新类会拥有基类的所有功能

- 当你需要创建多个类，它们拥有很多相似的成员变量或成员函数时，也可以使用继承。可以将这些类的共同成员提取出来，定义为父类，然后从父类继承，既可以节省代码，也方便后续修改成员

下面我们定义一个父类 People，然后定义了子类Student

```c++
#include<iostream>

using namespace std;

//父类（基类） Pelple
class People {
private:
    char *name;
    int age;

public:
    void setName(char *name) {
        this->name = name;
    }

    char *getName() {
        return this->name;
    }

    void setAge(int age) {
        this->age = age;
    }

    int getAge() {
        return this->age;
    }
};

//父类（派生类） Student
class Student : public People {
private:
    float score;

public:
    void setScore(float score) {
        this->score = score;
    }

    float getScore() {
        return this->score;
    }
};

//父类（派生类） Staff
class Staff : public People {
private:
    float money;

public:
    void setMoney(float money) {
        this->money = money;
    }

    float getMoney() {
        return this->money;
    }
};


int main() {

    // 创建Student学生对象
    Student boy;
    boy.setName("小明");
    boy.setAge(16);
    boy.setScore(95.5f);
    cout << boy.getName() << "的年龄是 " << boy.getAge() << "，成绩是 " << boy.getScore() << endl;

    // 创建Staff员工对象
    Staff girl;
    girl.setName("小丽");
    girl.setAge(18);
    girl.setMoney(4500.67);
    cout << girl.getName() << "的年龄是 " << girl.getAge() << "，月工资是 " << girl.getMoney() << "\n";

    return 0;
}
```

运行结果：

小明的年龄是16，成绩是95.5

小丽的年龄是18，月工资是4500.67

说明：

- Student 类继承了 People 类的成员，同时还新增了自己的成员变量 score 和成员函数 setScore()、getScore()
- 继承过来的成员，可以通过子类对象访问，就像自己的一样

继承的一般语法为：

```c++
class 子类名:［继承方式］父类名{
  子类新增加的成员
};
```

继承方式包括 public（公有的）、private（私有的）和 protected（受保护的），此项是可选的，如果不写，那么默认为 private。

### 在子类的函数中调用父类的成员

1. 使用this调用

在子类的函数中，调用从父类继承而来的成员变量或成员函数，直接使用this

‵‵‵c++
#include <iostream>

using namespace std;


class Father {
public:
    int age;
    int money;

    void set_new_age_money(int age, int money) {
        this->age = age;
        this->money = money;
    }

    void display() {
        cout << "Father::display(), age=" << this->age << ", money=" << this->money << "\n";
    }

};

class Children : public Father {
public:

    void test() {
        cout << "Children::test()" << "\n";
        // 在子类的函数中，调用从父类继承而来的成员变量，直接使用this
        cout << "继承的成员变量age=" << this->age << ", 继承的money=" << this->money << "\n";
        // 在子类的函数中，调用从父类继承而来的成员函数，直接使用this
        this->display();
    }

};

int main() {

    Children boy;
    boy.set_new_age_money(33, 1000000);  // 直接调用继承的父类成员函数
    boy.test();

    return 0;
}
```

输出：

Children:test()

继承的成员变量age=33，继承的money=1000000

Father::display(), age=33, money=100000

### 重写

1. 是什么？

一句话：子类定义了与父类相同名字的函数，覆盖了父类的这个函数

2. 有什么用？

一句话：扩展或者重新编写功能

3. 示例

‵‵‵ c++
#include <iostream>

using namespace std;


class A {
public:

    void display() {
        cout << "A::display()" << "\n";
    }
};

class B : public A {
public:

    void display() {
        A::display();
        cout << "B::display()" << "\n";
    }

    void call_parent_display() {
        this->display(); //调用的是自己类里面的display()函数
        A::display();  // 调用被重写的父类函数
    }
};

int main() {
    class B b;
    b.display();  // 调用子类的函数
    cout << "--------------------\n";
    b.call_parent_display();  // 调用子类中的函数，然后调用被重写的父类函数 

    return 0;
}
```

运行结果：

A：：display()

B：：display()

--------------

A：：display()

B：：display() % this->display

A：：display()

4. 注意

被重写的父类成员函数，无论是否有重载，子类中都不会继承，即在子类重写父类的函数后，在子类调用都不会再调用父类的函数了

```c++
#include <iostream>

using namespace std;


class A {
public:

    void display() {
        cout << "A::display()" << "\n";
    }

    void display(int num) {
        cout << "A::display(int num)" << "\n";
    }
};

class B : public A {
public:

    void display() {
        cout << "B::display()" << "\n";
        // this->display(100);  // 编译失败 
        A::display(100);
    }

};

int main() {
    class B b;
    b.display();  // 调用子类的函数
    // b.display(100);  // 编译失败 （无法再调用父类函数）

    return 0;
}
```

结果：

B：：display()

A::display（int num）


### 多层继承

1. 是什么？

就是父类还有父类，父类的父类还有父类......

例如你爷爷有1000w，那么你爸爸就继承过来了，同样的道理 你也可以从你爸爸那里继承得到这笔钱

2. 示例


‵‵‵c++
#include <iostream>

using namespace std;

class Grandpa {
public:
    void jiejian() {  // 节俭
        cout << "节俭" << "\n";
    }
};

class Father : public Grandpa {
public:
    void makeMoeny() {
        cout << "赚钱" << endl;
    }
};


class Son : public Father {

};

int main() {

    Son s;
    s.jiejian();
    s.makeMoeny();

    return 0;
}
‵‵‵

运行结果：

节俭

赚钱

-----


```c++
#include <iostream>

using namespace std;


class GrandPa {
public:
    void display() {
        cout << "GrandPa::display()\n";
    }

    void display_3() {
        cout << "GrandPa::display_3()\n";
    }
};

class Father : public GrandPa {
public:
    void display() {
        cout << "Father::display()\n";
    }

    void display_2() {
        cout << "Father::display_2()\n";
    }

};

class Children : public Father {
public:
    void display() {
        cout << "Children::display()" << "\n";
        // 调用父类的成员函数
        this->display_2();
        // 调用被重写的父类成员函数
        Father::display();
        // 调用爷爷类中的成员函数
        this->display_3();
        // 调用被重写的爷爷类中的成员函数
        GrandPa::display();
    }

};

int main() {

    Children boy;
    boy.display();

    return 0;
}
```

运行结果：

Children::display()

Father::display_2()

Father::display()

GrandPa::display_3()

GrandPa::display()

### 多重继承

1. 是什么？

C++ 允许存在多继承，也就是一个子类可以同时继承多个父类

2. 示例
```c++
#include <iostream>

using namespace std;

class Father {
public:
    void make_money() {
        cout << "赚钱" << endl;
    }
};

class Mother {
public:
    void make_homework() {
        cout << "做好菜" << endl;
    }
};

class Son : public Father, public Mother {

};

int main() {

    Son s;
    s.make_money();
    s.make_homework();

    return 0;
}
```

结果：

赚钱

做好菜

3. 多个父类相同的函数名

注意：要在子类中重写这个函数，否则会出现编译错误，原因是二义性

‵‵‵c++

#include <iostream>

using namespace std;

class Father {
public:
    void make_money() {
        cout << "Father类的 make_money" << endl;
    }
};

class Mother {
public:
    void make_homework() {
        cout << "Mother类的 make_homework" << endl;
    }

    void make_money() {
        cout << "Mother类的 make_money" << endl;
    }
};

class Son : public Father, public Mother {
public:
    // 如果子类不重写make_money函数，会导致s.make_money()在编译时失败，因为存在二义性
    void make_money() {
        Father::make_money();
        Mother::make_money();
    }
};

int main() {

    Son s;
    s.make_money();
    s.make_homework();

    return 0;
}

```
结果：

Father类的make_money

Mother类的make_money

Father类的make_homework


### 多态

1. 是什么？

一句话：有多个相同名字的函数，当调用时，会根据调用时的方式不同，调用不同的函数

2. 静态多态

以下代码，是一个有2个相同名字，但参数不同的函数组成，这样就形成了重载

当调用函数时，到底用哪个要根据调用时的参数而确定

```c++
#include <iostream>

using namespace std;

int Add(int a, int b) {
    cout << "int类型的函数被调用\n";
    return a + b;
}

double Add(double a, double b) {
    cout << "double类型的函数被调用\n";
    return a + b;
}

int main() {
    Add(10, 20);
    Add(10.0, 20.0);
    return 0;
}
```

运行结果：

int类型性的函数被调用

double类型的函数被调用

说明：

上述的Add函数，在编译阶段就已经确定了，因为编译器会根据实参的类型自动确定调用哪个函数，所以叫做静态多态

3. 动态多态

看下面的代码，f->show()到底调用哪个类中的show函数？

‵‵‵c++
#include<iostream>

using namespace std;

class Father {
public:
    void show() {
        cout << "father show" << endl;
    }
};

class Children : public Father {
public:
    void show() {
        cout << "children  show" << endl;
    }
};

int main() {
    Father *father = new Father();
    father->show();  // 调用父类的show函数

    Children *children = new Children();
    children->show();  // 调用子类的show函数

    Father *p = new Children();
    p->show();  // 调用哪个类中的show函数? 调用的是Father的

    return 0;
}
```

运行效果

father show

child show

father show

###  Virtual

1. 是什么？

可以让一个函数成为虚函数

2. 有什么用？

通过virtual可以实现真正的多态

虚函数可以在父类的指针指向子类对象的前提下，通过父类的指针调用子类的成员函数

这种技术让父类的指针或引用具备了多种形态，这就是所谓的多态

最终形成的功能：

- 如果父类指针指向的是一个父类对象，则调用父类的函数

- 如果父类指针指向的是一个子类对象，则调用子类的函数

3. 怎样用？

定义虚函数非常简单，只需要在函数声明前，加上 virtual 关键字即可

注意：

在父类的函数上添加 virtual 关键字，可使子类的同名函数也变成虚函数

```c++
#include<iostream>

using namespace std;

class Father {
public:
    virtual void show() {
        cout << "father show" << endl;
    }
};

class Children : public Father {
public:
    virtual void show() {
        cout << "children  show" << endl;
    }
};

int main() {
    Father *father = new Father();
    father->show();  // 调用父类的show函数

    Children *children = new Children();
    children->show();  // 调用子类的show函数

    Father *p = new Children();
    p->show();  // 如果父类指针指向的是一个子类对象，则调用子类的函数

    return 0;
}
```

结果：

Father show

Children show

Children show

```c++
#include <iostream>

using namespace std;

class WashMachine {
public:
    virtual void wash() {
        cout << "洗衣机在洗衣服" << endl;
    }
};


class SmartWashMachine : public WashMachine {
public:
    virtual void wash() {
        cout << "智能洗衣机在洗衣服" << endl;
    }
};


int main() {
    // 父类指针指向子类对象
    WashMachine *w2 = new SmartWashMachine();
    w2->wash();
    return 0;
}
```
结果

智能洗衣机在洗衣服


###  1. const

1. 是什么？

在C++中，const 是一种关键字，用于指定常量或表示不可修改

2. 使用const

2.1 修饰变量

声明常量：通过在变量声明前加上const关键字，可以将变量声明为常量，即其值不能被修改。

```c++
const int x = 5;  // 声明一个常量x，其值为5，不可修改
```

2.2 修饰函数参数

函数参数为只读：将函数参数声明为const表示函数内部不会修改该参数的值。

```c++
void func(const int x) {
    // x 是只读参数，不可修改
}
```

2.3 修饰函数返回值
返回值为只读：将函数返回值声明为const表示返回值为只读，不能被修改。

示例1
```c++
const int getValue() {
    return 42;
}
```

实例：
```c++
#include <iostream>

using namespace std;

class A {
private:
    int num;
public:
    A() : num(2) {}

    void set_num() {
        this->num = 10;
    }

    void get_num() const {
        printf("%d\\n", num);
    }
};

class B {
public:
    const class A *get() {
        class A *p = new A();
        return p;
    }
};

int main() {
    B b;
    b.get()->get_num();
    b.get()->set_num();  // 编译失败，因为返回的对象是const修饰，所以不允许修改成员
    return 0;
}
```

2.4 修饰类成员

常量成员变量：在类中使用 const 声明的成员变量是类的常量成员，一旦初始化后就不能修改

常量成员函数：在类中使用 const 修饰的成员函数表示该函数不会修改类的成员变量

```c++
class MyClass {
private:
    const int value;  // 声明一个常量成员变量

public:
    int getValue() const {
        return value;  // 常量成员函数，不能修改非静态成员变量
    }
};
```

2.5 修饰指针

const在*右侧：表示指针本身是一个常量，即指针指向的内存地址不能修改。

```c++
int x = 5;
int* const ptr = &x;  // ptr 是一个指向整型变量的常量指针
*ptr = 10;  // 可以修改指针指向的值
ptr++;  // 错误，指针本身是一个常量，不能修改指向的地址
```

const在 *左边：表示指针指向的值是一个常量，即指针所指向的内存地址的值不能修改。

```c++
int x = 5;
const int* ptr = &x;  // ptr 是一个指向常量整型的指针
*ptr = 10;  // 错误，不能修改指针指向的值
ptr++;  // 可以修改指针指向的地址
```

const在*左右都有：表示指针本身是常量，且指针指向的对象也是常量.

```c++
const int x = 5;
const int* const ptr = &x;  // ptr 是一个指向常量整型的常量指针
*ptr = 10;  // 错误，不能通过指针修改 x 的值
ptr++;  // 错误，指针本身和指向的对象都是常量
```

总结：

使用 const 的好处：

- 提高代码的可读性：通过明确标识常量和只读的对象，使代码更易于理解和维护。

- 防止意外修改：将对象声明为 const 可以防止在代码中无意中修改其值，提高代码的稳定性和可靠性。

- 编译器优化：编译器可以使用 const 信息进行优化，提高代码的执行效率。

需要注意的是，const 对象必须在声明时进行初始化，并且其值在编译时就确定了，不能在运行时修改。

在编写代码时，合理使用 const 可以提高代码的可读性、可靠性和性能。通过将对象声明为 const，可以明确指示其只读特性，并减少意外的修改，从而提高代码的质量和可维护性。

### 引用

1. 引言
   
引用是 C++ 中的一个重要特性，它提供了一种直接访问对象的方式，类似于变量的别名。引用提供了一种简洁、直观的方法来操作变量，能够提高代码的可读性和效率。

在本教程中，我们将探讨引用的基本概念、用法和注意事项。我们将看到引用作为函数参数和返回值的使用方式，了解引用和常量、数组、结构体/类之间的关系。我们还将讨论引用的生命周期和应用场景，以及如何避免悬空引用。

2. 引用的基本概念
   
2.1 引用的定义

引用是一个已存在对象的别名，它用于在代码中引用和操作该对象。引用的定义使用 & 符号，并在变量类型前加上 &。

```c++
int num = 10;
int& ref = num;  // 引用变量 ref 是变量 num 的别名
```

在上述示例中，我们声明了一个整型变量 num，然后通过 int& ref = num 将 ref 声明为 num 的引用。此时，ref 就成为了 num 的别名，对 ref 的操作实际上是对 num 的操作

2.2 引用与指针的比较

- 引用和指针是 C++ 中的两种不同的概念。它们都提供了对对象的间接访问方式，但在语法和语义上有一些区别。

- 引用是一个别名，一旦初始化后不可改变，总是引用同一个对象，而指针可以改变所指向的对象。

- 引用不需要使用解引用操作符（*）来访问所引用的对象，而指针需要使用解引用操作符来访问所指向的对象。

- 引用在声明时必须初始化，并且不能引用空值（NULL），而指针可以在声明后再进行初始化，并且可以指向空值。

- 引用不需要进行内存分配和释放，而指针需要手动进行内存管理。

2.3 引用的特性

- 引用没有独立的存储空间，它只是变量的别名，与原始变量共享同一块内存。

- 对引用的操作等效于对原始对象的操作，对引用的修改会直接反映到原始对象上。

- 引用可以用于函数参数传递和返回值，允许直接操作原始对象而不是复制对象。

- 引用可以提高代码的可读性，使代码更加直观和简洁。

3. 引用的用法
   
3.1 引用作为函数参数

引用常用于函数参数传递，允许在函数中直接操作原始对象，而不是复制对象的副本。通过使用引用参数，可以避免对象复制的开销，并使函数对原始对象的修改能够在函数外部可见。

```c++
void modifyValue(int& value) {
    value = 10;
}

int main() {
    int num = 5;
    modifyValue(num);  // 通过引用修改原始对象
    // 现在 num 的值为 10
    return 0;
}
```
在上述示例中，我们定义了一个函数 modifyValue，它接受一个整型引用参数 value。通过引用参数，我们可以直接修改原始对象 num 的值。

3.2 引用作为函数返回值

引用还可以作为函数的返回值，允许函数返回对其他变量的引用。这样可以方便地在表达式中使用函数返回的引用，*并直接修改原始对象*。

```c++
int& getLarger(int& a, int& b) {
    return (a > b) ? a : b;
}

int main() {
    int num1 = 5;
    int num2 = 10;
    int& largerNum = getLarger(num1, num2);  // 获取较大值的引用
    largerNum = 15;  // 直接修改原始对象
    // 现在 num2 的值为 15
    return 0;
}
```

在上述示例中，我们定义了一个函数 getLarger，它接受两个整型引用参数 a 和 b。函数通过比较两个参数的值，返回较大值的引用。我们将返回的引用赋给 largerNum 变量，并直接修改原始对象 num2 的值。

3.3 引用作为函数参数和返回值的注意事项

引用作为函数参数传递时，函数可以修改原始对象的值，因此需要小心操作引用，确保不会意外修改原始对象。

引用作为函数返回值时，确保返回的引用所引用的对象在函数调用结束后仍然有效。避免返回局部对象的引用，因为局部对象在函数返回后会被销毁。

3.4 引用和常量

引用可以与常量一起使用，从而创建常量引用。常量引用在函数参数传递中很有用，以避免对原始对象进行修改。

```c++
void printValue(const int& value) {
    // 只读访问 value，不会修改原始对象
}

int main() {
    int num = 10;
    printValue(num);  // 通过常量引用传递参数
    return 0;
}
```

在上述示例中，我们定义了一个函数 printValue，它接受一个整型常量引用参数 value。通过使用常量引用，我们确保函数内部只能读取参数的值，而不会修改原始对象 num。

3.5 引用和数组

引用可以与数组一起使用，以引用数组的元素或作为数组的别名。

```c++
void printArray(const int (&arr)[5]) {
    for (int i = 0; i < 5; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
}

int main() {
    int nums[] = {1, 2, 3, 4, 5};
    printArray(nums);  // 通过引用传递数组
    return 0;
}
```
在上述示例中，我们定义了一个函数 printArray，它接受一个整型数组的引用参数。通过使用数组引用参数，我们可以在函数内部访问数组的元素，并在函数外部调用时避免了数组的复制。

3.6 引用和结构体/类

引用可以与结构体或类一起使用，以引用结构体或类的对象。

```c++
struct Point {
    int x;
    int y;
};

void printPoint(const Point& p) {
    cout << "x: " << p.x << ", y: " << p.y << endl;
}

int main() {
    Point pt = {10, 20};
    printPoint(pt);  // 通过引用传递结构体对象
    return 0;
}
```

在上述示例中，我们定义了一个结构体 Point，它包含两个整型成员变量 x 和 y。然后，我们定义了一个函数 printPoint，它接受一个 Point 结构体对象的引用参数。通过引用参数，我们可以在函数内部访问结构体对象的成员。

4. 引用的生命周期

4.1 引用的初始化和赋值

引用在声明时必须进行初始化，一旦初始化后，它将始终引用同一个对象。引用的初始化可以在声明时进行，也可以在后续赋值操作中进行。

```c++
int num = 10;
int& ref = num;  // 引用初始化为 num

int anotherNum = 20;
ref = anotherNum;  // 引用仍然引用 num，并将 num 的值修改为20
```

在上述示例中，我们声明了一个整型变量 num 和一个整型引用 ref，并将引用初始化为 num。后续的赋值操作 ref = anotherNum 不会改变引用所引用的对象，而是修改了原始对象 num 的值。

4.2 引用的作用域

引用的作用域与变量的作用域相同。引用在定义所在的作用域内有效，超出该作用域后，引用不再有效。

```c++
int main() {
    int num = 10;
    {
        int& ref = num;  // 引用在内部作用域中定义
        ref = 20;  // 修改 num 的值为 20
    }
    // 此处引用 ref 不再有效
    return 0;
}
```

4.3 避免悬空引用

引用必须始终引用一个有效的对象，否则会产生悬空引用。悬空引用指的是引用一个已被销毁的对象或不存在的对象，这将导致未定义的行为。因此，在使用引用时要特别注意，确保引用所引用的对象的生命周期正确管理。

```c++
int& getReference() {
    int num = 10;
    return num;  // 返回局部变量的引用，产生悬空引用
}

int main() {
    int& ref = getReference();
    // 此处引用 ref 是悬空引用，访问引用将导致未定义的行为
    return 0;
}
```

在上述示例中，我们定义了一个函数 getReference，它返回一个局部变量 num 的引用。然而，当函数返回后，局部变量 num 被销毁，引用 ref 变成了悬空引用，访问引用将导致未定义的行为。

5. 引用的应用场景

引用在 C++ 中有许多应用场景，下面是一些常见的应用场景：

5.1 通过引用修改函数参数

使用引用作为函数参数可以直接修改原始对象，而不是复制对象的副本。这在需要修改函数参数的情况下非常有用。

```c++
void modifyValue(int& value) {
    value = 10;
}

int main() {
    int num = 5;
    modifyValue(num);  // 通过引用修改原始对象
    // 现在 num 的值为 10
    return 0;
}
```

在上述示例中，通过引用将 num 传递给函数 modifyValue，函数直接修改了原始对象 num 的值。

5.2 函数返回引用的链式操作

返回引用的函数可以用于实现链式操作，使代码更加简洁和易读。

```c++
class Counter {
private:
    int count;

public:
    Counter(int start = 0) : count(start) {} //构造函数，将构造函数的参数 start 的值赋给成员变量 count

    Counter& increment() {
        count++;
        return *this;
    }

    int getCount() const {
        return count;
    }
};

int main() {
    Counter c;
    c.increment().increment().increment();
    cout << c.getCount();  // 输出 3
    return 0;
}
```
在上述示例中，increment 函数返回对 Counter 对象的引用，使得可以对同一个对象进行多次操作，从而实现链式操作。

5.3 使用引用提高性能

使用引用可以避免对象的复制，从而提高程序的性能。当对象较大时，通过引用传递参数比通过值传递参数更高效。

```c++
void processLargeObject(LargeObject& obj) {
    // 对大对象进行处理
}

int main() {
    LargeObject obj;
    processLargeObject(obj);  // 通过引用传递大对象，避免复制
    return 0;
}
```

在上述示例中，我们将大对象 obj 通过引用传递给函数 processLargeObject，避免了对象的复制，提高了程序的性能。

5.4 使用引用参数避免对象复制

当函数需要访问和修改对象的成员时，使用引用参数可以避免对对象进行复制。

```c++
void modifyObject(MyObject& obj) {
    obj.setValue(10);
}

int main() {
    MyObject obj;
    modifyObject(obj);  // 通过引用传递对象，避免复制
    return 0;
}
```

5.5 引用作为容器的元素类型

引用可以作为容器（如数组、向量）的元素类型，允许在容器中存储对其他对象的引用。

```c++
int num1 = 10;
int num2 = 20;
int& ref1 = num1;
int& ref2 = num2;

vector<int&> refContainer;
refContainer.push_back(ref1);
refContainer.push_back(ref2);

for (int& ref : refContainer) {
    cout << ref << " ";
}
```

在上述示例中，我们声明了两个整型变量 num1 和 num2，并创建了对它们的引用 ref1 和 ref2。然后，我们创建了一个存储整型引用的向量 refContainer，并将 ref1 和 ref2 添加到容器中。最后，我们使用范围基于循环来遍历容器并输出引用所引用的值。

6. 总结
   
引用是 C++ 中的一个强大特性，它提供了直接访问对象的方式，并允许在函数和类中操作原始对象。引用不仅可以提高代码的可读性和效率，还可以简化代码和实现一些高级功能。


引用可以作为容器（如数组、向量）的元素类型，允许在容器中存储对其他对象的引用。
在上述示例中，我们通过引用将对象 obj 传递给函数 modifyObject，函数可以直接访问和修改对象的成员，而无需复制对象。


### 容器
1. 是什么？

在C++中，容器是用于存储和管理数据的对象。容器提供了一种将多个元素组织在一起的方式，并提供了一系列操作来方便地访问、插入、删除和修改数据。C++标准库提供了许多不同类型的容器，每种容器都有其特定的功能和用途。

2. String容器
   
std::string 是 C++ 标准库中提供的字符串容器，它用于存储和操作字符串。std::string 提供了许多字符串操作的方法，使得在 C++ 中处理字符串变得更加方便和高效。

‵‵‵c++
#include <iostream>
#include <string>

int main() {
    // 创建一个空的 std::string 对象
    std::string str;

    // 使用赋值操作给 std::string 添加内容
    str = "Hello, ";
    str += "World!";

    // 访问和修改 std::string 的内容
    std::cout << "Length: " << str.length() << std::endl; // 输出: 13
    std::cout << "First character: " << str[0] << std::endl; // 输出: H

    // 使用迭代器遍历 std::string 的字符
    for (auto it = str.begin(); it != str.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;

    // 在 std::string 中查找子串
    std::size_t found = str.find("World");
    if (found != std::string::npos) {
        std::cout << "Substring 'World' found at position: " << found << std::endl;
    } else {
        std::cout << "Substring 'World' not found" << std::endl;
    }

    return 0;
}
```

- 1.头文件

```c++
#include <string>
using namespace std;
```

- 2.创建和初始化 std::string 对象：

‵‵‵c++
string str;  // 创建一个空字符串
string str1 = "Hello";  // 使用字符串字面值初始化
string str2("World");  // 使用字符串字面值或字符数组初始化
string str3(str2);  // 使用另一个字符串初始化
```

- 3.获取字符串长度：

```c++
int length = str.length();  // 获取字符串的长度
```

- 4.连接字符串：

```c++
string fullName = str1 + " " + str2;  // 使用 + 运算符连接字符串
```

- 5.访问和修改字符：

```c++
char ch = str1[0];  // 获取字符串中的单个字符
str1[0] = 'h';  // 修改字符串中的单个字符
```

- 6.比较字符串：


```c++
bool isEqual = (str1 == str2);  // 比较两个字符串是否相等
bool isLess = (str1 < str2);  // 比较两个字符串的大小关系
```

- 7.搜索子字符串：

```c++
size_t found = str1.find("lo");  // 查找子字符串的位置
if (found != string::npos) {
    // 子字符串存在于原字符串中
}
```

- 8.截取子字符串：

```c++
string subStr = str1.substr(2, 3);  // 从指定位置截取子字符串
```

- 9.插入和删除字符串：

```c++
str1.insert(2, "123");  // 在指定位置插入字符串
str1.erase(2, 3);  // 从指定位置删除字符
```

- 10.转换为 C 风格字符串：

```c++
const char* cstr = str1.c_str();  // 转换为以空字符结尾的 C 风格字符串
```

- 11.输入和输出字符串：

```c++
cout << str1 << endl;  // 输出字符串
cin >> str1;  // 输入字符串
```

3. Vector容器

std::vector 是 C++ 标准库中提供的动态数组容器，它能够存储和管理任意类型的元素。std::vector 提供了方便的方法来操作数组，使得在 C++ 中处理动态数组变得更加灵活和高效。

```c++
#include <iostream>
#include <vector>

int main() {
    // 创建一个空的 std::vector 对象
    std::vector<int> vec;

    // 向 std::vector 添加元素
    vec.push_back(10);
    vec.push_back(20);
    vec.push_back(30);

    // 使用索引操作符访问 std::vector 的元素
    std::cout << "First element: " << vec[0] << std::endl;  // 输出: 10
    std::cout << "Second element: " << vec[1] << std::endl;  // 输出: 20

    // 使用迭代器遍历 std::vector 的元素
    for (auto it = vec.begin(); it != vec.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;

    // 在 std::vector 中插入元素
    vec.insert(vec.begin() + 1, 15);  // 在索引为1的位置插入元素15

    // 在 std::vector 中删除元素
    vec.erase(vec.begin() + 2);  // 删除索引为2的元素

    // 检查 std::vector 是否为空
    if (vec.empty()) {
        std::cout << "Vector is empty" << std::endl;
    } else {
        std::cout << "Vector is not empty" << std::endl;
    }

    // 访问 std::vector 中的元素数量
    std::cout << "Vector size: " << vec.size() << std::endl;

    return 0;
}
```

以下是关于 std::vector 的详细教程：

- 1.包含头文件：

```c++
#include <vector>
using namespace std;
```

- 2.创建和初始化 std::vector 对象：

```c++
vector<int> nums;  // 创建一个空的整型向量
vector<int> nums1 = {1, 2, 3};  // 使用初始化列表初始化向量
vector<int> nums2(5, 0);  // 创建包含5个元素，每个元素初始化为0的向量
vector<int> nums3(nums1);  // 使用另一个向量初始化向量
```

- 3.获取向量的大小：

```c++
int size = nums.size();  // 获取向量中的元素个数
```

- 4.访问和修改元素：

```c++
int value = nums[0];  // 通过下标访问元素
nums[0] = 10;  // 修改元素的值

int frontValue = nums.front();  // 获取首个元素的值
int backValue = nums.back();  // 获取最后一个元素的值
```

- 5.在向量末尾添加元素：

```c++
nums.push_back(5);  // 在向量末尾添加一个元素
```

- 6.在指定位置插入和删除元素：

```c++
nums.insert(nums.begin() + 2, 8);  // 在指定位置插入一个元素
nums.erase(nums.begin() + 1);  // 删除指定位置的元素
```

- 7.迭代遍历向量：

```c++
// 普通的方式
for (int i = 0; i < nums.size(); i++) {
    cout << nums[i] << " ";
}

// 新的方式
for (int num : nums) {
    cout << num << " ";
}
```

- 8.清空向量：

```c++
nums.clear();  // 清空向量中的所有元素
```

- 9.检查向量是否为空：

```c++
bool isEmpty = nums.empty();  // 检查向量是否为空
```

- 10.动态调整向量的大小：

```c++
nums.resize(10);  // 调整向量的大小为10，新增元素使用默认值
nums.resize(5, 0);  // 调整向量的大小为5，新增元素初始化为0
```

- 11.排序向量中的元素：

```c++
sort(nums.begin(), nums.end());  // 对向量中的元素进行升序排序
```

4. 其它常用容器

4.1 std::array

功能：*固定*大小的数组，提供高效的随机访问，大小在编译时确定。

头文件：#include <array>

用法示例：

```c++
#include <iostream>
#include <array>

int main() {
    std::array<int, 5> myArray = {1, 2, 3, 4, 5};

    // 访问元素
    std::cout << "First element: " << myArray[0] << std::endl;
    std::cout << "Second element: " << myArray.at(1) << std::endl;
    std::cout << "Last element: " << myArray.back() << std::endl;

    // 修改元素
    myArray[2] = 10;

    // 下标遍历
    for (int i = 0; i < myArray.size(); i++) {
        std::cout << myArray[i] << " ";  // 使用下标访问元素
    }
    std::cout << "\n----------------\n";

    // 迭代遍历
    for (const auto &element: myArray) {
        std::cout << element << " ";
    }
    std::cout << "\n----------------\n";

    // 迭代器操作
    for (auto it = myArray.begin(); it != myArray.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

5.2 std::set
功能：有序集合，存储唯一值，支持快速的插入、查找和删除操作。

特点：元素按照严格的排序顺序进行存储，即每个元素都会根据其值进行排序。

头文件：#include <set>

用法示例：

```c++
#include <iostream>
#include <set>

int main() {
    std::set<int> mySet;

    // 插入元素到 set
    mySet.insert(10);
    mySet.insert(20);
    mySet.insert(30);
    mySet.insert(20);  // 重复元素将被忽略

    // 遍历 set 并打印元素
    for (const auto &element: mySet) {
        std::cout << element << " ";
    }
    std::cout << "\n";

    // 查找元素
    auto it = mySet.find(20);
    if (it != mySet.end()) {
        std::cout << "Element 20 found in set" << std::endl;
    } else {
        std::cout << "Element 20 not found in set" << std::endl;
    }

    // 删除元素
    mySet.erase(20);

    // 检查 set 是否为空
    if (mySet.empty()) {
        std::cout << "Set is empty" << std::endl;
    } else {
        std::cout << "Set is not empty" << std::endl;
    }

    // 访问 set 中的元素数量
    std::cout << "Set size: " << mySet.size() << std::endl;

    return 0;
}
```
5.3 std::map
功能：关联数组（字典），按键值对存储和访问数据。(设定字符串对应数字)

头文件：#include <map>

用法示例：

```c++
#include <map>
#include "iostream"

using namespace std;

int main() {
    map<string, int> scores;  // 创建空的字符串-整型映射
    scores["Alice"] = 90;  // 添加键值对
    scores["Bob"] = 85;
    scores["Charlie"] = 95;

    for (std::pair<string, int> xx: scores) {
        cout << xx.first << ": " << xx.second << endl;
    }

    cout << "----------------------------\n";

    for (const auto &pair: scores) {
        cout << pair.first << ": " << pair.second << endl;
    }
    
    return 0;
}
```

5.4 std::stack
功能：栈，遵循后进先出（LIFO）的原则，只能在栈顶进行插入和删除操作。

头文件：#include <stack>

用法示例：

```c++
#include <iostream>
#include <stack>

int main() {
    std::stack<int> myStack;

    // 添加元素到堆栈
    myStack.push(10);
    myStack.push(20);
    myStack.push(30);

    // 访问和移除堆栈顶部的元素
    std::cout << "Top element: " << myStack.top() << std::endl;  // 输出: 30
    myStack.pop();
    std::cout << "Top element: " << myStack.top() << std::endl;  // 输出: 20

    // 检查堆栈是否为空
    if (myStack.empty()) {
        std::cout << "Stack is empty" << std::endl;
    } else {
        std::cout << "Stack is not empty" << std::endl;
    }

    // 访问堆栈中的元素数量
    std::cout << "Stack size: " << myStack.size() << std::endl;

    return 0;
}
```

5.5 std::queue
功能：队列，遵循先进先出（FIFO）的原则，只能在队尾进行插入，在队头进行删除操作。

头文件：#include <queue>

用法示例：

```c++
#include <iostream>
#include <queue>

int main() {
    std::queue<int> myQueue;

    // 添加元素到队列
    myQueue.push(10);
    myQueue.push(20);
    myQueue.push(30);

    // 访问和移除队列的元素
    std::cout << "Front: " << myQueue.front() << std::endl;  // 输出: 10
    myQueue.pop();
    std::cout << "Front: " << myQueue.front() << std::endl;  // 输出: 20

    // 检查队列是否为空
    if (myQueue.empty()) {
        std::cout << "Queue is empty" << std::endl;
    } else {
        std::cout << "Queue is not empty" << std::endl;
    }

    // 访问队列中的元素数量
    std::cout << "Queue size: " << myQueue.size() << std::endl;  // 输出: 2

    return 0;
}
```
