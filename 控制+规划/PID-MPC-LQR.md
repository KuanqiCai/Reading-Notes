# 一、PID 

## 1 基本概念:

- 基本表达式：

  连续形式
  $$
  u(t)=K_p*e(t)+K_i*\int e(t)dt +K_d*\frac{de(t)}{dt}
  $$

  - 比例系数Kp:

    能提高系统的动态响应速度，迅速反映误差，从而减少误差，但是不能消除误差，简单来说就是越大越快越小越慢但是可能会超调或者过慢有很多弊端，并且太大了会不稳定。

  - 积分系数Ki:

    一般就是消除稳态误差，只要系统存在误差积分作用就会不断积累，输出控制量来消除误差，如果偏差为零这时积分才停止，但是积分作用太强会使得超调量加大，甚至使系统出现震荡

  - 微分系数Kd:

    三个参数中的预备人员，一般不用，在反馈量噪声比较大时可能会使系统震荡。Kd增大可以加快系统响应，减小超调量，适用于迟滞系统或无阻尼系统。微分控制是一种提前控制，以偏差的变化率为基准进行控制。

- 实际使用：

  在计算机控制系统中，由于控制是使用采样控制，它只能根据采样时刻的偏差计算控制量，而不能像模拟控制那样连续输出控制量量，进行连续控制。所以数字PID控制也属于离散型控制系统。(注重点：实际使用pid控制器是离散控制系统)

### 1.2 位置式PID:

pid第一种离散形式：

位置式PID控制的输出与整个过去的状态有关，用到了误差的累加值，输出绝对控制量。

- 公式：

  由于是离散型控制系统，积分和微分部分必须离散化处理。离散化处理的方法为：以 T 作为采样周期，k 作为采样序号，则离散采样时间 kT 对应着连续时间t，用矩形法数值积分近似代替积分，用一阶后向差分近似代替微分：
  $$
  u_k=Kp\big[e_k+ \frac{T}{Ti}\sum_{j=0}^ke_j+Td\frac{e_k-e_{k-1}}{T} \big] \tag{1}
  $$

- 特点：

  由于全量输出，所以每次输出均与过去状态有关，计算时要对ek进行累加，工作量大，耗内存；并且，因为计算机输出的uk 对应的是执行机构的实际位置，如果计算机出现故障，输出的 uk将大幅度变化，会引起执行机构的大幅度变化，有可能因此造成严重的生产事故

- Arduino 示例

  控制input从10到100

  ```c
  int setpoint = 100;
  long sumerror;
  double kp = 0.3,ki = 0.15,kd = 0.1;
  int lasterror;
  int input = 10;
  int output;
  int nowerror;
  void setup() {
    // put your setup code here, to run once:
      Serial.begin(9600);
  }
  
  void loop() {
    // put your main code here, to run repeatedly:
    nowerror = setpoint - input;
    sumerror += nowerror; 
    output = kp*nowerror + ki*sumerror + kd*(lasterror - nowerror);
    lasterror = nowerror;
    input += output;
    Serial.println(input);
    delay(50); 
   
  }
  ```

### 1.3 增量式PID:

pid第二种离散形式

Increment PID Control：增量式PID通过对控制量的增量（本次控制量和上次控制量的差值）进行控制，输出两个时刻的控制量变化$\Delta u_k$。

当执行机构需要的控制量是增量，而不是位置量的绝对数值时，可以使用增量式PID控制算法进行控制。

- 公式：

  由位置式PID公式推导得出。

  k-1时刻的控制量为:
  $$
  u_{k-1}=Kp\big[e_{k-1}+ \frac{T}{Ti}\sum_{j=0}^{k-1}e_j+Td\frac{e_{k-1}-e_{k-2}}{T} \big] \tag{2}
  $$
  1式 - 2式可得增量输出
  $$
  \begin{align}
  \Delta U(k)&=U(k)-U(k-1)\\
  		&=K_p[e_k-e_{k-1}]+K_i[e_k]+K_d[e_k-2e_{k-1}+e_{k-2}]
  
  \end{align}
  $$

- 特点：

  增量式 PID 控制算法与位置式 PID 算法公式相比，如果计算机控制系统采用恒定的采样周期 T，一旦确定Kp、Ti、Td参数，只要使用前后三次测量的偏差值，就可以由增量式PID公式求出控制量。计算量小的多，因此在实际中得到广泛的应用。

- Arduino 例子

  控制Input从10到100

  ```python
  int setpoint = 100;
  long SUMERROR;
  double kp = 0.3,ki = 0.15,kd = 0.1;
  int LASTERROR;
  int PREVERROR;
  int input = 10;
  int output;
  int ERROR;
  void setup() {
    // put your setup code here, to run once:
      Serial.begin(9600);
  }
  
  void loop() {
    // put your main code here, to run repeatedly:
    ERROR = setpoint - input;
    SUMERROR += ERROR;
    output = kp*ERROR + ki*SUMERROR + kd*(LASTERROR - PREVERROR);
    PREVERROR =  LASTERROR;
    LASTERROR = ERROR;
    input +=output;
    Serial.println(input);
    delay(50);  
  }
  ```

  

# 二、MPC

[参考](https://zhuanlan.zhihu.com/p/99409532)

Model Predictive Control模型预测控制实际上是以优化方法来求解控制问题。

# 三、LQR

[参考](https://zhuanlan.zhihu.com/p/139145957)

Linear Quadratic Regulator线性二次型调节器

# 四、一些控制概念
## 1. 波特率，比特率，通信速度
- **波特率**：表示每秒钟传送码元符号的个数，是衡量数据传送速率的指标。  
  - 在信息传输通道中，携带数据信息的信号单元叫码元，每秒钟通过信道传输的码元数称为码元传输速率，简称波特率。  
  - 波特(Baud)调制速率，单位时间内载波调制状态变化的次数。波特可理解为是一个传输单元。 
  - 1波特即指每秒传输1个码元符号，通过不同的调制方式，可以在一个码元符号上负载多个bit位信息。
- **比特率**：每秒传送的比特(bit)数。单位为 bps(Bit Per Second)，比特率越高，每秒传送数据就越多。1字节(Byte)=8比特(bit)
- **传输速率**：传输速率是一种泛指，指的是数据从一点向另一点传输的速率。包含上面的比特率、波特率等。
- **通信速度**：通信速度和传速速度同样也是一种泛指。比如I²C通信速度100KHz，SPI最大通信速度支持150Mbps。
- **波特率VS比特率**：
  - 比特率=波特率x单个调制状态对应的二进制位数： 1 Baud = log2M(bit/s).   
  其中M是信号的编码级数。
  - 可以得出：
    - 两相调制(单个调制状态对应1个二进制位)的比特率等于波特率；
    - 四相调制(单个调制状态对应2个二进制位)的比特率为波特率的两倍；
    - 八相调制(单个调制状态对应3个二进制位)的比特率为波特率的三倍；依次类推。
- **举例**：
  - 串口9600波特率每秒可传输多少字节？
    - 两相调制时，比特率为9600bps
    - 传输1个数据需要：起始位:1,数据位:2,停止位:1,校验位:0 共10bit
    - 所以美妙传输9600/10 = 960字节
  - I2C通信速度100KHz什么意思：
    - I2C属于同步通信，有一根时钟线(SCL)，100KHZ即这个时钟线的频率
