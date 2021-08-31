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
  
    