- 字符串

  C语言中，字符串实际上是使用 **null** 字符 **\0** 终止的一维字符数组。

  ```C
  char site[7] = {'R', 'U', 'N', 'O', 'O', 'B', '\0'};
  char site[] = "RUNOOB";
  ```

- 定义数组大小

  ```c
  /*错误：error array index is not constant
  因为，在C语言中const不代表constan，只是说明这个是read-only
  */
  int const int x = 5;
  int arr[x] = {1, 2, 3, 4, 5};
  
  //正确1：
  #define x 5
  int arr[x] = {1, 2, 3, 4, 5};
  //正确2:
  enum { LEN = 5 };
  int arr[LEN] = {1, 2, 3, 4, 5};
  ```

- 把int变量赋值给char数组

  ```c
  char p[10];
  int i =0;
  p[0]=(char)('0'+i);
  ```

- C语言中无bool类型定义，需要自己写

  ```c
  typedef enum
  {
      true=1, false=0
  }bool;
  ```

- 结构Struct

  ```c
  //用typedef可以当一个数据类型来用
  typedef struct{
      int day;
      char *month;
      int year;
  }date;
  date today ={4,"may",2021};
  
  today.day = 4
      ..
  ```

- 位域Bit-fields(1)

  存储某些信息不需要整个字节时使用

  ```c
  # define BIT0 01
  # define BIT1 02
  # define BIT2 04
  
  //或者
  enum {BIT0 = 01, BIT1 = 02, BIT2 = 04};
  ```

## 文件

https://blog.csdn.net/afei__/article/details/81835684

- 打开关闭一个文件

```c
//declare a file
File *fp;

//open or close a file
/*
r:reading the file from the beginning; returns an errorif the file does not exist
w:writing to the file from the beginning; creates the file if it does not exist
a:appending to the end of the file; creates the file if it does not exist.
*/
FILE *fp = fopen (" hello . txt", "w");
fclose(fp);
```

- 读取写入

  C语言fprintf()和fscanf()函数

  - `fprintf()`函数用于将一组字符写入文件。它将格式化的输出发送到流。

  ```c
  int fprintf(FILE *stream, const char *format [, argument, ...])
  ```

  - `fscanf()`函数用于从文件中读取一组字符。它从文件读取一个单词，并在文件结尾返回`EOF`。
  - 打开文件要用r模式，不然无法正确读取。
  
  ```c
  int fscanf(FILE *stream, const char *format [, argument, ...])
  ```
  
  - 例子
  
    ```c
    #include <stdio.h>  
    main() {
        FILE *fp;
        fp = fopen("file.txt", "w");//opening file  
        fprintf(fp, "Hello file by fprintf...\n");//writing data into file  
        fclose(fp);//closing file  
        printf("Write to file : file.txt finished.");
        
        char buff[255];//creating char array to store data of file  
      	fp = fopen("file.txt", "r");  
      	while(fscanf(fp, "%s", buff)!=EOF){  
       		printf("%s ", buff );  
        }  
        fclose(fp);  
    }
    ```
  
    

# C++实践课

1. **Was ist der Unterschied zwischen der dynamischen und statischen Speicherreservierung?**

   | dynamischen                            | statischen                                |
   | -------------------------------------- | ----------------------------------------- |
   | zur Laufzeit Speicher reserviert       | Vor Laufzeit(Compile) Speicher reserviert |
   | Die Speicher wird im Laufzeit befreit. | Die Speicher wird nach Laufzeit befreit.  |

   C中用malloc()和free()

   c++中用new和 delete，new 相比malloc更强因为他会同时构造对象

   ```c++
   // Dynamische Speicherreservierung für eine Variable vom Typ double
   double* pdWidth = new double;
   // Dynamische Speicherreservierung für ein Array der Dimension 80 vom Typ char 
   char* pchName = new char[80];
   // Freigeben einer dynamisch angelegten Variable
   delete pdWidth; 
   // Freigeben eines dynamisch angelegten Arrays 
   delete[] pchName;
   ```

   

2. **Wann benutzt man dynamische Speicherreservierung?**

   Wenn es vor der Laufzeit eines Programms nicht bekannt ist, wie viel Speicher für die Programmausführung notwendig ist. 

3. **Welche Gefahren gehen von Pointern aus?**

   Pointer können auch noch existieren, wenn die referenzierte Variable gelöscht ist. Greift man dann auf den Pointer zu, dereferenziert man undefinierten Speicherbereich, was unweigerlich zu einem Fehler führt.

   ```c++
   // 指针的用法1：
   // Zugreifen auf den ersten Buchstaben des Labels 
   char chLetter = *pchLabel; 
   chLetter = pchLabel[0]; 
   chLetter = pchLabel[]; 
   // Zugreifen auf den fünften Buchstaben 
   chLetter = *(pchLabel+4); 
   chLetter = pchLabel[4];
   
   // 指针的用法2：
   // Signatur der Funktion, die den Pointer auf ein Array mit Nutzdaten 
   // erhält, die den Mittelwert daraus berechnet 
   void calculateMean(float* pfMean, int* piData, int iSize); 
   。。。
   int main() {
       float fMean = 0.0; 
       int piData[] = {1, 2, 3, 4, 58, 5, 6, 1, 98, 3}; 
       // Aufruf der Funktion 
       calculateMean(&fMean, piData, 10); 
       return 0;
   }
   ```

   

4. **Welche Vorteile haben Referenzen引用 gegenüber Pointern?**

   - Der Vorteil gegenüber Pointern besteht im Sicherheitsgewinn.

   - Ein weiterer Vorteil gegenüber Pointern liegt in der vereinfachten Parameterübergabe beim Aufruf von Funktionen

     ```c++
     // 引用的用法1：
     // Initialisierung einer Variablen
     int iStudentID = 27002; 
     // Initialisierung auf die Variable
     int& riStudentID = iStudentID;
     
     // 引用的用法2：
     // Definition einer Funktion die eine Referenz und eine Variable als  
     // Uebergabeparameter erwartet 
     void changeStudentID(int& riOldStudentID, int iNewStudentID){ 
         riOldStudentID = iNewStudentID;
     }
     // Die Funktion wird wie folgt in der Main aufgerufen
     int main(){
         int iStudentIDDetlefCruse = 99933;
         changeStudentID(iStudentIDDetlefCruse, 999773);
         return 0;
     }
     ```

     