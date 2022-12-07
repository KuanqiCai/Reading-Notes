# 零、学习资源整理

1. [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/index.html)
2. [本笔记所学习的课](https://github.com/huggingface/deep-rl-class)
3. [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2020.pdf)(已保存在onedrive)
4. 李宏毅深度强化学习：
   - [笔记](https://github.com/changliang5811/leedeeprl-notes)
   - [课程](http://speech.ee.ntu.edu.tw/~tlkagk/courses_MLDS18.html)，[bilibili](https://www.bilibili.com/video/BV1MW411w79n/)

# 一、深度强化学习

网课：https://github.com/huggingface/deep-rl-class#the-hugging-face-deep-reinforcement-learning-class-

## 1. Foundation基础

- Formal Definition

  Reinforcement learning is a framework for solving control tasks (also called decision problems) by building agents that learn from the environment by interacting with it through trial and error and receiving rewards (positive or negative) as unique feedback

### 1.1 强化学习中心思想

强化学习的中心思想，就是**让智能体在环境里学习**。每个行动会对应各自的奖励，智能体通过分析数据来学习，怎样的情况下应该做怎样的事情。

比如智能体要学着玩超级马里奥：

![](https://picd.zhimg.com/80/v2-e058a9d05a30d09edbbd0b70da685133_720w.webp?source=1940ef5c)

- 智能体在环境 (超级马里奥) 里获得初始状态**S0** (游戏的第一帧) ；

- 在S0的基础上，agent会做出第一个行动**A0** (如向右走) ；

- 环境变化，获得新的状态**S1** (A0发生后的某一帧) ；

- 环境给出了第一个奖励**R1** (没死：+1) ；

  于是，这个[loop](https://www.zhihu.com/search?q=loop&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A602826860})输出的就是一个**由状态、奖励和行动组成的序列**。

  而智能体的**目标**就是让**预期累积奖励最大化**。





### 1.2 奖励假说reward hypothesis

由上可知目标是要让累积奖励最大化

- 每一个时间步的累积奖励可以表示为：
  $$
  G_t = R_{t+1}+R_{t+2}+...
  $$
  但现实中**不能把奖励直接相加**。因为游戏里，越接近游戏开始处的奖励，就越容易获得；而随着游戏的进行，后面的奖励就没有那么容易拿到了。

- 给奖励添加折扣

  用$\gamma$表示折扣率，0-1之间

  - $\gamma$越大，折扣越小。表示智能体越在意长期的奖励。
  - $\gamma$越小，折扣越大。表示智能体越在意短期的奖励。

  $$
  G_t = R_{t+1} + \gamma R_{t+2}+\gamma^2 R_{t+3}+...
  $$

  简单来说，**离困难的地方近一步**，**就乘上一个γ**，表示奖励越难获得。





### 1.3 强化学习的两种任务

#### 1.3.1 片段性任务Episodic Task

这类任务，有个**起点**，有个**终点**。两者之间有一堆状态，一堆行动，一堆奖励，和一堆新的状态，它们共同构成了一“集”。当一集结束，也就是到达终止状态的时候，智能体会看一下奖励累积了多少，以此**评估自己的表现**。

然后，它就带着之前的经验开始一局新游戏。这一次，智能体做决定的依据会充分一些。

比如马里奥游戏：

- 永远从同一个起点开始
- 如果马里奥死了或者到达终点，则游戏结束
- 结束时得到一系列状态、行动、奖励和新状态
- 算出奖励的总和 (看看表现如何)
- 更有经验地开始新游戏

**集数越多**，**智能体的表现会越好**



#### 1.3.2 连续性任务Continuing Task

**永远不会有游戏结束的时候**。智能体要学习如何选择最佳的行动，和环境进行实时交互。就像自动驾驶汽车，并没有过关拔旗子的事。

这样的任务是通过时间**差分学习** (Temporal Difference Learning) 来训练的。每一个时间步，都会有总结学习，等不到一集结束再分析结果。



### 1.4 探索和开发之间的权衡

- **探索** (Exploration) 是找到关于环境的更多信息：

  You go every day to the same one that you know is good and **take the risk to miss another better restaurant.**

- **开发** (Exploitation) 是利用已知信息来得到最多的奖励：

  Try restaurants you never went to before, with the risk of having a bad experience **but the probable opportunity of a fantastic experience.**

我们需要设定一种规则，让智能体能够**把握二者之间的平衡**，来处理以下困境：

- 老鼠在迷宫里吃奶酪，在它附近可以迟到无数快分散的奶酪(+1)，迷宫深处有巨型奶酪(+1000)但也存在危险。
  - 如果我们只关注吃了多少，老鼠只会在附近安全处慢慢的吃，永远不会去迷宫深处。
  - 如果它选择探索跑去迷宫深处，也许能发现大奖但也可能遇到危险。

### 1.5 观测和状态空间Observations/States Space

- Observations/States are the **information our agent gets from the environment.** 

  - In the case of a video game, it can be a frame (a screenshot). 

  - In the case of the trading agent, it can be the value of a certain stock, etc.

- 区别：

  - *State s*: is **a complete description of the state of the world** (there is no hidden information). In a fully observed environment.
  - *Observation o*: is a **partial description of the state.** In a partially observed environment.

### 1.6 动作空间Action Space

- The Action space is the set of **all possible actions in an environment.**

- The actions can come from a *discrete* or *continuous space*:

  - *Discrete space*: the number of possible actions is **finite**.
    比如马里奥游戏只有4个方向的跳远

  - *Continuous space*: the number of possible actions is **infinite**.

    比如自动驾驶无数方向的运动

### 1.7 强化学习的三种方法：

记住：RL的目标是找到一个optimal policy π.有下面3种方法，但现实中只用前2种

![](https://github.com/Fernweh-yang/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/Reinforcement%20Learning/two-approaches%20to%20find%20policy.jpg?raw=true)

#### 1.7.1 基于价值(value-based)

- 目标是**优化价值函数V(s)**: 价值函数value function会告诉我们，智能体在每个状态里得出的未来奖励最大预期 (maximum expected future reward) 。In Value-based methods, instead of training a policy function, **we train a value function that maps a state to the expected value of being at that state**.

- 有两种基于价值的方法：

  1. The State-Value function
     $$
     V_\pi(s) = E_\pi[R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+... | S_t=s]
     $$
     For each state, the state-value function outputs the expected return if the agent **starts at that state,** and then follow the policy forever after (for all future timesteps if you prefer).

     - E: Expected
     - S: State
     - R：reward

  2. The Action-Value function
     $$
     Q_\pi(s,a) = E_\pi[R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+... | S_t=s,A_t=a]
     $$
     In the Action-value function, for each state and action pair, the action-value function **outputs the expected return** if the agent starts in that state and takes action, and then follows the policy forever after.

  3. Difference:

     - In state-value function, we calculate **the value of a state $S_t$**
     - In action-value function, we calculate **the value of the state-action pair $(S_t,A_t)$** hence the value of taking that action at that state.

- 所以基于价值的方法：

  - don't train the policy
  - The policy is a function defined by hand.

- Value 和 Policy之间的联系
  $$
  \pi^*(s)=arg\ \underset{a}{max}Q^*(s,a)
  $$
  

#### 1.7.2 基于策略 (policy-based)

- The Policy π **is the brain of our Agent**, it’s the function that tell us what action to take given the state we are. So it defines the agent’s behavior at a given time.
- This policy function will **map from each state to the best corresponding action at that state**. Or a **probability distribution over the set of possible actions at that state**.

- 目标是 **优化策略函数$\pi(s)$** , 策略就是评判智能体在特定时间点的表现。把每一个状态和它所对应的最佳行动建立联系。

  策略分为两种

  - **确定性**策略：某一个特定状态下的策略，永远都会给出同样的行动。


  - **随机性**策略：策略给出的是多种行动的可能性分布。
    $$
    Stochastic\ Policy:\pi(a|s)=P[A_t=a|S_t=s]
    $$

    - S: State
    - A: Action


- 所以基于策略的方法
  - Train directly the policy
  - The policy is a Neural Network.

#### 1.7.3 基于模型(model-based)

这种方法是对环境建模。这表示，我们要创建一个模型，来表示环境的行为。

问题是，**每个环境**都会需要一个不同的模型 (马里奥每走一步，都会有一个新环境) 。这也是这个方法在强化学习中并不太常用的原因。

#### 1.7.4 总结对比

- Consequently, whatever method you use to solve your problem, **you will have a policy**, but in the case of value-based methods you don't train it, your policy **is just a simple function that you specify** (for instance greedy policy) and this policy **uses the values given by the value-function to select its actions.**
- So the difference is:
  - In policy-based, **the optimal policy is found by training the policy directly.**
  - In value-based, **finding an optimal value function leads to having an optimal policy.**

## 2. Q-Learning

Q-learning是一种基于价值的算法。The **Q comes from "the Quality" of that action at that state.**

### 2.1 贝尔曼方程

Bellman Equation: simplify our value estimation

1.7.1中提到的2种基于价值的函数(state-value or action-value function)，都需要sum all the rewards an agent can get if it starts at that state.这就非常的tedious冗长的。因此用贝尔曼方程来简化价值函数的计算。
$$
V_\pi(s)=E_\pi [R_{t+1}+\gamma V_\pi(S_{t+1})|S_t=s]
$$

- $R_{t+1}$: immediate reward
- $\gamma V_\pi(S_{t+1})$:  discounted value of the next state
-  the idea of the Bellman equation is that instead of calculating each value as the sum of the expected return, **which is a long process**
-  

### 2.2 两种训练价值函数的策略strategies

- RL agent都是通过使用之前的经验来学习，于此他们的区别是：
  - Monte Carlo uses **an entire episode of experience before learning**
  - Temporal Difference uses **only a step$(S_t,A_t,R_{t+1},S_{t+1})$ to learn**

#### 2.2.1蒙特卡洛(Monte Carlo)

Monte Carlo waits until the end of the episode, calculates $G_t$(expected return) and uses it as **a target for updating $V(S_t)$.**So it requires a **complete entire episode of interaction before updating our value function.**

在玩完一局后才总结学习。

- 公式：
  $$
  V(S_t)\leftarrow V(S_t)+\alpha[G_t-V(S_t)]
  $$

  - $V(S_t)$: value of state t

    - 左侧的那个是：新的状态价值
    - 右侧的2个是：之前估计的estimate的状态价值

  - $G_t$: sum of the total rewards at timestep

- 流程：

  1. 在一个episode关卡结束后，我们得到这一局一系列的state,action,reward和new state
  2. agent将所有reward相加，得到$G_t$
  3. 利用上面的公式更新价值函数$V(S_t)$
  4. 用新学到的知识开始新一轮的游戏。

#### 2.2.2时间差分(Temporal Difference)

Temporal difference, on the other hand, waits for only one interaction (one step)$S_{t+1}$ to form a TD target and update $V(S_t)$ using $R_{t+1}$ and $\gamma V(S_{t+1})$。

每走一步学习一次。

- 公式：
  $$
  V(S_t) \leftarrow V(S_t) + \alpha[\underbrace{\underbrace{R_{t+1}+\gamma V(S_{t+1})}_{TD\ Target}-V(S_t)}_{TD\ Error}]
  $$

  - $V(S_t)$: value of state 
    - 左侧的那个是：新的状态价值
    - 右侧的2个是：之前估计的estimate的状态价值
  - $V(S_{t+1})$: value of next state
  - $\gamma$: discount
  - $R_{t+1}$: reward

- 流程

  1. 每一步结束后，我们会得到state,action,reward和new state

  2. 将$V(S_{t+1})$和$R_{t+1}$用于estimate $G_t$,并利用上式更新新的价值函数$V(S_t)$

     

### 2.3 [Q-Learning](https://huggingface.co/blog/deep-rl-q-part2#what-is-q-learning)是什么

- Q-Learning is **the algorithm we use to train our Q-Function**, an action-value function that determines the value of being at a particular state and taking a specific action at that state.

- Q-Learning是一个off-policy, value-based函数并使用Temporal Difference方法来训练它的action-value function.

- Q-Learning是一个用来训练Q-function的算法。

  ![](https://github.com/Fernweh-yang/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/Reinforcement%20Learning/Q-function.jpg?raw=true)

  - Q-function是一个action-value function。它determine了在一个particular state和采取specific action时的value。
    - If we have an optimal Q-function, we have an optimal policy since we know for each state what is the best action to take.
  - Q-table记录了所有的state-action pair values ,是Q-function的memory。
    - 在给予了一个action和state后Q-function会搜索这个Q-table并输出一个值。
    - 一开始Q-table通常会全都初始为0，在explore环境时更新Q-table

### 2.4 [Q-Learning算法](https://huggingface.co/blog/deep-rl-q-part2#the-q-learning-algorithm)

![](https://github.com/Fernweh-yang/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/Reinforcement%20Learning/Q-learning-Algorithm.jpg?raw=true)

1. 第一步：初始化Q-Table的每一个state-action pair为0

2. 第二步：使用**Epsilon Greedy Strategy**来选择action

   Epsilon Greedy Strategy is a policy that handles the exploration/exploitation trade-off.

   - 定义一个探索率epsilon $\epsilon=1.0$,即随机执行某一个action的概率（do exploration）。刚开始学习时，这个速率必须是最高值，因为我们对Q-table的取值一无所知。这意味着我们需要通过随机选择我们的行动进行大量探索。
   - $1-\epsilon$则是开发exploitation的概率，根据当前的Q-table信息选择 action with the highest state-action pair value

3. 第三步：perform执行action $A_t$  得到reward $R_{t+1}$和 下一个state$S_{t+1}$

4. 第四步：更新state-action pair $Q(S_t,A_t)$

   - 因为Q-Learning是时间差分算法TD，所以根据2.2.2,用如下公式更新
     $$
     Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha[\underbrace{\underbrace{R_{t+1}+\gamma\ \underset{a}{max} Q(S_{t+1},a)}_{TD\ Target}-Q(S_t,A_t)}_{TD\ Error}]\\
     $$

   - $\underset{a}{max} Q(S_{t+1},a)$: 这里更新算法时，我们总是选择带来highest state-action value的动作, 所以用到的是**greedy policy**而不是第二步中用到的 epsilon greedy policy。因为epsilon greeedy policy只有在一个随机数大于$\epsilon$时才原则最大值。

### 2.5 对比：off-policy和on-policy

- off-policy:using **a different policy for acting and updating.**
  - Q-Learning中用greedy policy来更新，用epsilon greedy policy来行动
- on-policy:using the **same policy for acting and updating.**
  - 另一个value-based的强化学习算法：sarsa,就是更新和行动都使用epsilon greedy policy





## 3. Deep Q-learning

### 3.1 与Q-learning的对比

- 传统的Q-learning在需要巨大的state space问题时，需要花大量时间生成和更新Q-table，所以传统的Q-Learning变得ineffective。为了解决大state space的问题，Deep Q-learning不会使用Q-table，而是由一个神经网络获取状态并根据状态为每一个动作近似一个Q-value。
- Deep Q-Learning **uses a deep neural network to approximate the different Q-values for each possible action at a state** (value-function estimation).

![](https://github.com/Fernweh-yang/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/Reinforcement%20Learning/deep%20q%20and%20q%20learning.jpg?raw=true)



### 3.2 Deep Q-Network(DQN)

![](https://github.com/Fernweh-yang/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/Reinforcement%20Learning/deep-q-network.jpg?raw=true)

- **Preprocess the input** is an essential step since we want to reduce the complexity of our state.
  - 比如上面这个像素游戏，
    - 可以grayscale将3通道rgb转为1个通道，因为颜色在这个游戏环境中不重要。
    - 还可以将分辨率从160X210压缩到84X84.
    - 还可以crop一部分游戏屏幕，因为不是所有地方都是有用信息

- **temporal limitation**时间限制
  - 只有1frame帧图(一个时刻)，我们不知道小球的运动方向，但如果将多帧放在一起看，我们就可以知道：啊，小球在向右上方飞。
  - 所以为了得到temporal information时间信息，we stack four frames together.
- stacked frames会被3个convolutional layers处理
  - These layers **allow us to capture and exploit spatial relationships in images**
  - Because frames are stacked together, **you can exploit some spatial properties across those frames**.

- 最后经过全连接层，得到某1个state下每一个可能动作的Q-value

### 3.3 Deep Q-Learning Algorithm

相比于2.4Q-Learning算法的第四步，Deep Q-learning使用一个Loss function比较 predicted Q-value和Q-target。然后用gradient descent来更新Q-Networks的权重，如此反复得以更好的近似Q-values

Q-Target: 
$$
y_j=r_j+\gamma max_{a'}\hat{Q}(\phi_{j+1},a';\theta^-)
$$
Q-Loss:
$$
y_j-Q(\phi_j,a_j;\theta)
$$
对比普通Q-learning的 td-target(q-target)和 td-error(q-loss)
$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha[\underbrace{\underbrace{R_{t+1}+\gamma\ \underset{a}{max} Q(S_{t+1},a)}_{TD\ Target}-Q(S_t,A_t)}_{TD\ Error}]\\
$$


Deep Q-Learning算法有2个步骤

- **Sampling**: we perform actions and **store the observed experiences tuples元组 in a replay memory**.
- **Training**: Select the **small batch of tuple randomly and learn from it using a gradient descent update step**.

![](https://github.com/Fernweh-yang/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/Reinforcement%20Learning/deep%20q%20algorithm.jpg?raw=true)

由于引进了 Neural Network,相比Q-Learning更不稳定，所以需要加入如下3种措施

1. *Experience Replay*, to make more **efficient use of experiences**.
2. *Fixed Q-Target* **to stabilize the training**.
3. *Double Deep Q-Learning*, to **handle the problem of the overestimation高估 of Q-values**.

![](https://github.com/Fernweh-yang/Reading-Notes/blob/main/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/Reinforcement%20Learning/stable%20deep%20q.jpg?raw=true)

#### 3.3.1 Experience Replay

experience replay:初始化第一行，sampling的最后1行 和 training的第一行 

在Deep Q-learning中experience replay有2个功能

1. **Make more efficient use of the experiences during the training**.

   - Usually, in online reinforcement learning, we interact in the environment, get experiences (state, action, reward, and next state), learn from them (update the neural network) and discard them.
   - But with experience replay, we create a replay buffer缓冲 D that saves experience samples **that we can reuse during the training.**=》This allows us to **learn from individual experiences multiple times**.

2. **Avoid forgetting previous experiences and reduce the correlation相关性 between experiences**.

   - The problem we get if we give sequential samples of experiences to our neural network is that it tends to forget **the previous experiences as it overwrites new experiences.** For instance, if we are in the first level and then the second, which is different, our agent can forget how to behave and play in the first level.

   - This prevents **the network from only learning about what it has immediately done.**

   - 也避免了action values oscillate震荡和diverge发散

     

#### 3.3.2 Fixed Q-Target

- 当我们计算TD Error(Q-loss)时,我们计算**TD target (Q-Target)和current Q-value (estimation of Q)**的区别。

  但我们不知道真正的TD Target是什么，所以我们也需要用如下公式更新TD Target：
  $$
  y_j=r_j+\gamma max_{a'}\hat{Q}(\phi_{j+1},a';\theta^-)
  $$

- 但我们在更新TD Target和Q value时，用的是同一组parameter。这就导致在训练的每一步，我们的Q-Value在shif的同时，Q-target也在shift。就像是在追逐一个会动的目标。

  这会导致significant oscillation震荡 in training.

- 为了解决上述问题，就需要fixed Q-target:

  - Use a **separate network with a fixed parameter** for estimating the TD Target
  - **Copy the parameters from our Deep Q-Network at every C step** to update the target network.

#### 3.3.3 Double DQN

Double DQNs, or Double Learning 由[Hado van Hasselt](https://papers.nips.cc/paper/3964-double-q-learning)提出，解决了**overestimation高估 of Q-values**的问题。

- 我们在更新TD target时，要如何确认我们选择了**the best action for the next state is the action with the highest Q-value**？
  - The accuracy of Q values depends on what action we tried **and** what neighboring states we explored.
  - 因为一开始我们没有足够得分explore环境，所以我们选择最大Q Value时，可能是不准确的(noisy)，从而导致false positives。
  - 如果non-optimal actions are regularly **given a higher Q value than the optimal best action, the learning will be complicated.**
- 为了解决上述问题，需要使用2个神经网络来让 `动作action的选择` 和 `target Q value的生成` 解耦decouple:
  - Use our **DQN network** to select the best action to take for the next state (the action with the highest Q value).
  - Use our **Target network** to calculate the target Q value of taking that action at the next state.



## 4. Unity MLAgents

- [Unity ML-Agents toolkit](https://github.com/Unity-Technologies/ml-agents) is a plugin based on the game engine Unity that allows us to use the **Unity Game Engine as an environment builder to train agents.**

  [Unity ML-Agents Toolkit with hugging face](https://github.com/huggingface/ml-agents) 可以让我们不用下载unity直接使用mlagent

### 4.1 Four components

Unity ML-Agents has four essential components.

1. **Learning environment**:

    **contains the Unity scene (the environment) and the environment elements** (game characters)**.**

2. **Python API**

   contains the **low-level Python interface** for **interacting and manipulating the environment**. It’s the API we use to launch the training.

3. **communicator**

   connects the environment (C#) with the Python API (Python).

4. **Python trainers**

   the RL algorithms made with PyTorch (PPO, SAC…).

### 4.2 Inside the Learning Component

There are three important elements

1. Agent:

   the actor of the scene

2. Brain:

   the policy we optimize to train the agent

3. Academy:

   - This element **orchestrates策划 agents and their decision-making process.** 

   - 可以把academy想象成一个maestro音乐大师，负责处理来自python api的请求。比如python api说"we need some observations", academy就让agents去搜集一些observations。

   - The Academy **will be the one that will send the order to our Agents and ensure that agents are in sync**同步:

     - Collect Observations

     - Select your action using your policy

     - Take the Action

     - Reset if you reached the max step or if you’re done.

### 4.3 Pyramid Environment

我们这里用来训练的环境

- The goal in this environment is to train our agent **to get the gold brick金砖 on the top of the Pyramid金字塔.** 为了实现这个目标，我们需要：

  - a button to spawn a pyramid
  - navigate to the Pyramid
  - knock it over
  - move to the gold brick at the top

- reward system:

  - -0.001: for every step

    given this negative reward for every step, will push our agent to go faster

  - +2 for moving to golden brick

  - 为了训练agent能seeks that button and then the Pyramid to destroy，这里使用2种Rewards

    1. The *extrinsic外在 one* given by the environment.
    2. But also an *intrinsic内在 one* called *curiosity*. **This second will push our agent to be curious, or in other terms, to better explore its environment.** 关于Deep RL中的curiosity见副2

- Observation:

  - Instead of normal vision(frame), we use 148 raycasts光线投射 that can each detect objects (switch, bricks, golden brick, and walls.)
  - a **boolean variable indicating表明 the switch state** (did we turn on or not the switch to spawn the Pyramid) 
  -  a vector that **contains agent’ speed.**

- Action space:

  - Forward Motion: Up/Down
  - Rotation: Rotate left / Rotate right

## 5. Policy Gradient with Pytorch



# 二、Stable-baseline3实现一、的代码

## 1. Foundation:

### 1.1 gym 的基本用法：

```python
import gym

# First, we create our environment called LunarLander-v2
env = gym.make("LunarLander-v2")

# Then we reset this environment
observation = env.reset()

for _ in range(20):
  # Take a random action
  action = env.action_space.sample()
  print("Action taken:", action)

  # Do this action in the environment and get
  # next_state, reward, done and info
  observation, reward, done, info = env.step(action)
  
  # If the game is done (in our case we land, crashed or timeout)
  if done:
      # Reset the environment
      print("Environment is reset")
      observation = env.reset()
```

### 1.2 Stable-Baselines3的基本使用步骤

```python
# Create environment
env = gym.make('LunarLander-v2')

# Instantiate the agent
model = PPO('MlpPolicy', env, verbose=1)
# Train the agent
model.learn(total_timesteps=int(2e5))
```



### 1.3 一个自动降落的智能飞机

运行在colab：https://colab.research.google.com/github/huggingface/deep-rl-class/blob/main/unit1/unit1.ipynb

#### 1.3.1 安装用于colab虚拟界面显示

  ```python
  !sudo apt-get update
  !apt install python-opengl
  !apt install ffmpeg
  !apt install xvfb
  !pip3 install pyvirtualdisplay
  
  # Virtual display
  from pyvirtualdisplay import Display
  
  virtual_display = Display(visible=0, size=(1400, 900))
  virtual_display.start()
  ```

#### 1.3.2 **安装深度学习库**

  ```python
  !pip install importlib-metadata==4.12.0 # To overcome an issue with importlib-metadata https://stackoverflow.com/questions/73929564/entrypoints-object-has-no-attribute-get-digital-ocean
  !pip install gym[box2d]
  !pip install stable-baselines3[extra]
  !pip install huggingface_sb3
  !pip install pyglet==1.5.1
  !pip install ale-py==0.7.4 # To overcome an issue with gym (https://github.com/DLR-RM/stable-baselines3/issues/875)
  
  !pip install pickle5
  ```

#### 1.3.3 **导入包**

  ```python
  import gym
  
  from huggingface_sb3 import load_from_hub, package_to_hub, push_to_hub
  from huggingface_hub import notebook_login # To log to our Hugging Face account to be able to upload models to the Hub.
  
  from stable_baselines3 import PPO
  from stable_baselines3 import DQN	# 扩展部分：尝试使用另个算法DQN
  from stable_baselines3.common.evaluation import evaluate_policy
  from stable_baselines3.common.env_util import make_vec_env
  ```

#### 1.3.4 **建立环境**

- 环境文档：https://www.gymlibrary.dev/environments/box2d/lunar_lander/

  - Action Space:
    - do nothing
    - fire left orientation engine
    - fire main engine
    - fire right orientation engine.
  - Observation Space
    - the coordinates of the lander in `x` & `y`
    - its linear velocities in `x` & `y`
    - its angle
    - its angular velocity
    - two booleans that represent whether each leg is in contact with the ground or not.
  - Rewards
    - Moving from the top of the screen to the landing pad and zero speed is about 100~140 points.
    - Firing main engine is -0.3 each frame
    - Each leg ground contact is +10 points
    - Episode finishes if the lander crashes (additional - 100 points) or come to rest (+100 points)

-  建立并查看环境

```python
# 使用gym创建环境
env = gym.make("LunarLander-v2")
env.reset()	#将环境重置
print("_____OBSERVATION SPACE_____ \n")
print("Observation Space Shape", env.observation_space.shape)
print("Sample observation", env.observation_space.sample()) # Get a random observation
print("\n _____ACTION SPACE_____ \n")
print("Action Space Shape", env.action_space.n)
print("Action Space Sample", env.action_space.sample()) # Take a random action


# 使用stable baseline3向量化环境
# We create a vectorized environment (method for stacking multiple independent environments into a single environment) of 16 environments, this way, we'll have more diverse experiences during the training.
env = make_vec_env('LunarLander-v2', n_envs=16)
```



#### 1.3.5 **建立RL算法模型**

- 使用 [Stable Baselines3 (SB3)](https://stable-baselines3.readthedocs.io/en/master/)，SB3 is a set of **reliable implementations of reinforcement learning algorithms in PyTorch**.

##### 1.[PPO算法](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#example%5D)

```python
# We use MultiLayerPerceptron (MLPPolicy) because the input is a vector,
# if we had frames as input we would use CnnPolicy
model = PPO(
  policy = 'MlpPolicy',
  env = env,
  n_steps = 1024,
  batch_size = 64,
  n_epochs = 4,
  gamma = 0.999,
  gae_lambda = 0.98,
  ent_coef = 0.01,
  verbose=1)
```

##### 2.DQN算法

```python
model = DQN(
    policy='MlpPolicy',
    env = env,
    verbose=1,
    batch_size= 64  
)
```



#### 1.3.6 **训练模型**

1. PPO:

   ```python
   # Train it for 500,000 timesteps
   model.learn(total_timesteps=500000)
   # Save the model
   model_name = "ppo-LunarLander-v2"
   model.save(model_name)
   ```

   

2. DQN:

   ```python
   # Train it for 500,000 timesteps
   model.learn(total_timesteps=500000)
   # Save the model
   model_name = "dqn-LunarLander-v2"
   model.save(model_name)
   ```

   

#### 1.3.7 **评估训练完的智能体**

```python
eval_env = gym.make("LunarLander-v2")
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
```

#### 1.3.8 **发布模型到Huggingface上**

- [Huggingface库](https://github.com/huggingface/huggingface_sb3/tree/main#hugging-face--x-stable-baselines3-v20)功能：

  - You can **showcase our work** 🔥
  - You can **visualize your agent playing** 👀
  - You can **share with the community an agent that others can use** 💾
  - You can **access a leaderboard 🏆 to see how well your agent is performing compared to your classmates** 👉

- 连接电脑到Huggingface

  - Create a new token (https://huggingface.co/settings/tokens) **with write role**

    输入如下命令后会要求输入token

  ```python
  # 如果用的colab或者jupyter
  notebook_login()
  !git config --global credential.helper store
  # 如果用的本地python环境
  huggingface-cli login
  ```

- 使用package_to_hub()函数来上传代码

  ```python
  import gym
  from stable_baselines3.common.vec_env import DummyVecEnv
  from stable_baselines3.common.env_util import make_vec_env
  
  from huggingface_sb3 import package_to_hub
  
  ## TODO: Define a repo_id
  ## repo_id is the id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name} for instance ThomasSimonini/ppo-LunarLander-v2
  repo_id = "TUMxudashuai/ppo-LunarLander-v2"
  
  # TODO: Define the name of the environment
  env_id =  "LunarLander-v2"
  
  # Create the evaluation env
  eval_env = DummyVecEnv([lambda: gym.make(env_id)])
  
  
  # TODO: Define the model architecture we used
  model_architecture = "PPO"
  
  ## TODO: Define the commit message
  commit_message = "Upload PPO LunarLander-v2 trained agent"
  
  # method save, evaluate, generate a model card and record a replay video of your agent before pushing the repo to the hub
  package_to_hub(model=model, # Our trained model
                 model_name=model_name, # The name of our trained model 
                 model_architecture=model_architecture, # The model architecture we used: in our case PPO
                 env_id=env_id, # Name of the environment
                 eval_env=eval_env, # Evaluation Environment
                 repo_id=repo_id, # id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name} for instance ThomasSimonini/ppo-LunarLander-v2
                 commit_message=commit_message)
  
  # Note: if after running the package_to_hub function and it gives an issue of rebasing, please run the following code
  # cd <path_to_repo> && git add . && git commit -m "Add message" && git pull 
  # And don't forget to do a "git push" at the end to push the change to the hub.
  ```

  - model here: https://huggingface.co/TUMxudashuai/ppo-LunarLander-v2
    - see a video preview of your agent at the right.
    - click "Files and versions" to see all the files in the repository.
    - click "Use in stable-baselines3" to get a code snippet that shows how to load the model.
    - a model card (`README.md` file) which gives a description of the model

#### 扩展：

1. Try different hyperparameters of `PPO`. You can see them at https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#parameters.
2. Check the [Stable-Baselines3 documentation](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html) and try another models such as DQN.
3. Try to **change the environment**, why not using CartPole-v1, MountainCar-v0 or CarRacing-v0? Check how they works [using the gym documentation](https://www.gymlibrary.dev/) and have fun

### 1.4 倒立摆cartpole

#### 1.4.1 环境

```python
!sudo apt-get update
!apt install python-opengl
!apt install ffmpeg
!apt install xvfb
!pip3 install pyvirtualdisplay

# Virtual display
from pyvirtualdisplay import Display

virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

!pip install importlib-metadata==4.13.0
!pip install gym[classic_control]
!pip install stable-baselines3[extra]
!pip install huggingface_sb3
!pip install pyglet==1.5.1
!pip install ale-py==0.7.4 # To overcome an issue with gym (https://github.com/DLR-RM/stable-baselines3/issues/875)

!pip install pickle5
```

#### 1.4.2 加载库

```python
import gym

from huggingface_sb3 import load_from_hub, package_to_hub, push_to_hub
from huggingface_hub import notebook_login # To log to our Hugging Face account to be able to upload models to the Hub.

from stable_baselines3 import PPO
from stable_baselines3 import DQN	# 扩展部分：尝试使用另个算法DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
```

#### 1.4.3 创建环境

```python
env = gym.make('CartPole-v1')
env.reset()
print("_____OBSERVATION SPACE_____ \n")
print("Observation Space Shape", env.observation_space.shape)
print("Sample observation", env.observation_space.sample()) # Get a random observation
print("\n _____ACTION SPACE_____ \n")
print("Action Space Shape", env.action_space.n)
print("Action Space Sample", env.action_space.sample()) # Take a random action
env = make_vec_env('CartPole-v1', n_envs=16)
```

#### 1.4.4 创建模型

```python
model = PPO(
  policy = 'MlpPolicy',
  env = env,
  n_steps = 1024,
  batch_size = 64,
  n_epochs = 4,
  gamma = 0.999,
  gae_lambda = 0.98,
  ent_coef = 0.01,
  verbose=1)
```



#### 1.4.5 训练模型

```
# Train it for 500,000 timesteps
model.learn(total_timesteps=500000)
model_name = "ppo-CartPole-v1"
model.save(model_name)
```



## 2. Q-Learning

代码地址：https://colab.research.google.com/github/huggingface/deep-rl-class/blob/main/unit2/unit2.ipynb

### 2.1 Frozen Lake

#### 2.1.1 安装环境

```python
!sudo apt-get update%%capture
!pip install pyglet==1.5.1 
!apt install python-opengl
!apt install ffmpeg
!apt install xvfb
!pip3 install pyvirtualdisplay

# Virtual display
from pyvirtualdisplay import Display

virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

%%capture
!pip install gym==0.24 # We install the newest gym version for the Taxi-v3 "rgb_array version"
!pip install pygame
!pip install numpy

!pip install huggingface_hub
!pip install pickle5
!pip install pyyaml==6.0 # avoid key error metadata
!pip install imageio imageio_ffmpeg
```

#### 2.1.2 导入包

```python
import numpy as np
import gym
import random
import imageio
import os 

import pickle5 as pickle
from tqdm.notebook import tqdm
```

#### 2.1.3 建立环境

- 环境信息：https://www.gymlibrary.dev/environments/toy_text/frozen_lake/
  - 目标是： train our Q-Learning agent **to navigate from the starting state (S) to the goal state (G) by walking only on frozen tiles (F) and avoid holes (H)**.
  - 有两个尺寸的地图：
    - `map_name="4x4"`: a 4x4 grid version
      - 有16种可能的observations
    - `map_name="8x8"`: a 8x8 grid version
      - 有64种可能的observations
  - 环境有2种模式：
    - `is_slippery=False`: The agent always move in the intended打算的 direction due to the non-slippery nature of the frozen lake.
    - `is_slippery=True`: The agent may not always move in the intended direction due to the slippery nature of the frozen lake (stochastic随机的).
  - 有四种可能的行动，action space
    - 0: GO LEFT
    - 1: GO DOWN
    - 2: GO RIGHT
    - 3: GO UP
  - Reward function
    - Reach goal: +1
    - Reach hole: 0
    - Reach frozen: 0

- 代码

  ```python
  env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
  env.reset()
  print("_____OBSERVATION SPACE_____ \n")
  print("Observation Space", env.observation_space)
  print("Sample observation", env.observation_space.sample()) # Get a random observation
  print("\n _____ACTION SPACE_____ \n")
  print("Action Space Shape", env.action_space.n)
  print("Action Space Sample", env.action_space.sample()) # Take a random action
  ```

#### 2.1.4 创建并初始化Q-table

由一、2.4的第一步可知

Q-table是行row为state，列column为action的一个表格

```python
# openAI gym 提供如下两个参数告知状态空间和动作空间的维度
state_space = env.observation_space.n
print("There are ", state_space, " possible states")
action_space = env.action_space.n
print("There are ", action_space, " possible actions")

# 将Q-table初始化为0
def initialize_q_table(state_space, action_space):
  Qtable = np.zeros((state_space, action_space))
  return Qtable
Qtable_frozenlake = initialize_q_table(state_space, action_space)
```



#### 2.1.5 定义epsilon-greedy policy

由一、2.4的第二步,第三步可知

```python
def epsilon_greedy_policy(Qtable, state, epsilon):
  # Randomly generate a number between 0 and 1
  random_int = random.uniform(0,1)
  # if random_int > greater than epsilon --> exploitation
  if random_int > epsilon:
    # Take the action with the highest value given a state
    # np.max：接受一个参数，返回数组中的最大值；
    # np.argmax：接受一个参数，返回数组中最大值对应的索引；
    action = np.argmax(Qtable[state])
  # else --> exploration
  else:
    # 下面这个函数可以Sample a random action from the entire action space 
    action = env.action_space.sample()
  
  return action
```

#### 2.1.6 定义greedy policy

由一、2.4的第四步可知

```python
def greedy_policy(Qtable, state):
  # Exploitation: take the action with the highest state, action value
  action = np.argmax(Qtable[state])
  
  return action
```

#### 2.1.7 定义hyperparameters

定义2.4中公式以及训练的参数

```python
# Training parameters
n_training_episodes = 10000  # Total training episodes
learning_rate = 0.7          # Learning rate

# Evaluation parameters
n_eval_episodes = 100        # Total number of test episodes

# Environment parameters
env_id = "FrozenLake-v1"     # Name of the environment
max_steps = 99               # Max steps per episode
gamma = 0.95                 # Discounting rate
eval_seed = []               # The evaluation seed of the environment

# Exploration parameters
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.05            # Minimum exploration probability 
decay_rate = 0.0005            # Exponential decay rate for exploration prob
```



#### 2.1.8 建立训练循环

```python
def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
  for episode in tqdm(range(n_training_episodes)):
    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
    # Reset the environment
    state = env.reset()
    step = 0
    done = False

    # repeat
    for step in range(max_steps):
      # Choose the action At using epsilon greedy policy
      action = epsilon_greedy_policy(Qtable, state, epsilon)

      # Take action At and observe Rt+1 and St+1
      # Take the action (a) and observe the outcome state(s') and reward (r)
      new_state, reward, done, info = env.step(action)

      # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
      Qtable[state][action] = Qtable[state][action] + learning_rate * (reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action])   

      # If done, finish the episode
      if done:
        break
      
      # Our state is the new state
      state = new_state
  return Qtable
```



#### 2.1.9 训练模型

```python
# 训练模型
Qtable_frozenlake = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable_frozenlake)

# 查看训练完后的Q-table
Qtable_frozenlake
```



#### 2.1.10 评估模型

```python
def evaluate_agent(env, max_steps, n_eval_episodes, Q, seed):
  """
  Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
  :param env: The evaluation environment
  :param n_eval_episodes: Number of episode to evaluate the agent
  :param Q: The Q-table
  :param seed: The evaluation seed array (for taxi-v3)
  """
  episode_rewards = []
  for episode in tqdm(range(n_eval_episodes)):
    if seed:
      state = env.reset(seed=seed[episode])
    else:
      state = env.reset()
    step = 0
    done = False
    total_rewards_ep = 0
    
    for step in range(max_steps):
      # Take the action (index) that have the maximum expected future reward given that state
      action = np.argmax(Q[state][:])
      new_state, reward, done, info = env.step(action)
      total_rewards_ep += reward
        
      if done:
        break
      state = new_state
    episode_rewards.append(total_rewards_ep)
  mean_reward = np.mean(episode_rewards)
  std_reward = np.std(episode_rewards)

  return mean_reward, std_reward

mean_reward, std_reward = evaluate_agent(env, max_steps, n_eval_episodes, Qtable_frozenlake, eval_seed)
print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
```

#### 2.1.11 发布模型到Huggingface

1. 不修改下面的代码，用于上传

```python
%%capture
from huggingface_hub import HfApi, HfFolder, Repository
from huggingface_hub.repocard import metadata_eval_result, metadata_save

from pathlib import Path
import datetime
import json

#-------------------------------------------------------------------------------
def record_video(env, Qtable, out_directory, fps=1):
  images = []  
  done = False
  state = env.reset(seed=random.randint(0,500))
  img = env.render(mode='rgb_array')
  images.append(img)
  while not done:
    # Take the action (index) that have the maximum expected future reward given that state
    action = np.argmax(Qtable[state][:])
    state, reward, done, info = env.step(action) # We directly put next_state = state for recording logic
    img = env.render(mode='rgb_array')
    images.append(img)
  imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps)

#-------------------------------------------------------------------------------
def push_to_hub(repo_id, 
                model,
                env,
                video_fps=1,
                local_repo_path="hub",
                commit_message="Push Q-Learning agent to Hub",
                token= None
                ):
  _, repo_name = repo_id.split("/")

  eval_env = env
  
  # Step 1: Clone or create the repo
  # Create the repo (or clone its content if it's nonempty)
  api = HfApi()
  
  repo_url = api.create_repo(
        repo_id=repo_id,
        token=token,
        private=False,
        exist_ok=True,)
  
  # Git pull
  repo_local_path = Path(local_repo_path) / repo_name
  repo = Repository(repo_local_path, clone_from=repo_url, use_auth_token=True)
  repo.git_pull()
  
  repo.lfs_track(["*.mp4"])

  # Step 1: Save the model
  if env.spec.kwargs.get("map_name"):
    model["map_name"] = env.spec.kwargs.get("map_name")
    if env.spec.kwargs.get("is_slippery", "") == False:
      model["slippery"] = False

  print(model)
  
    
  # Pickle the model
  with open(Path(repo_local_path)/'q-learning.pkl', 'wb') as f:
    pickle.dump(model, f)
  
  # Step 2: Evaluate the model and build JSON
  mean_reward, std_reward = evaluate_agent(eval_env, model["max_steps"], model["n_eval_episodes"], model["qtable"], model["eval_seed"])

  # First get datetime
  eval_datetime = datetime.datetime.now()
  eval_form_datetime = eval_datetime.isoformat()

  evaluate_data = {
        "env_id": model["env_id"], 
        "mean_reward": mean_reward,
        "n_eval_episodes": model["n_eval_episodes"],
        "eval_datetime": eval_form_datetime,
  }
  # Write a JSON file
  with open(Path(repo_local_path) / "results.json", "w") as outfile:
      json.dump(evaluate_data, outfile)

  # Step 3: Create the model card
  # Env id
  env_name = model["env_id"]
  if env.spec.kwargs.get("map_name"):
    env_name += "-" + env.spec.kwargs.get("map_name")

  if env.spec.kwargs.get("is_slippery", "") == False:
    env_name += "-" + "no_slippery"

  metadata = {}
  metadata["tags"] = [
        env_name,
        "q-learning",
        "reinforcement-learning",
        "custom-implementation"
    ]

  # Add metrics
  eval = metadata_eval_result(
      model_pretty_name=repo_name,
      task_pretty_name="reinforcement-learning",
      task_id="reinforcement-learning",
      metrics_pretty_name="mean_reward",
      metrics_id="mean_reward",
      metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
      dataset_pretty_name=env_name,
      dataset_id=env_name,
    )

  # Merges both dictionaries
  metadata = {**metadata, **eval}

  model_card = f"""
  # **Q-Learning** Agent playing **{env_id}**
  This is a trained model of a **Q-Learning** agent playing **{env_id}** .
  """

  model_card += """
  ## Usage
  """

  model_card += f"""model = load_from_hub(repo_id="{repo_id}", filename="q-learning.pkl")

  # Don't forget to check if you need to add additional attributes (is_slippery=False etc)
  env = gym.make(model["env_id"])

  evaluate_agent(env, model["max_steps"], model["n_eval_episodes"], model["qtable"], model["eval_seed"])
  """

  model_card +=""" """

  readme_path = repo_local_path / "README.md"
  readme = ""
  if readme_path.exists():
      with readme_path.open("r", encoding="utf8") as f:
        readme = f.read()
  else:
    readme = model_card

  with readme_path.open("w", encoding="utf-8") as f:
    f.write(readme)

  # Save our metrics to Readme metadata
  metadata_save(readme_path, metadata)

  # Step 4: Record a video
  video_path =  repo_local_path / "replay.mp4"
  record_video(env, model["qtable"], video_path, video_fps)

  # Push everything to hub
  print(f"Pushing repo {repo_name} to the Hugging Face Hub")
  repo.push_to_hub(commit_message=commit_message)

  print(f"Your model is pushed to the hub. You can view your model here: {repo_url}")
```

2. 上传到自己的账号

   https://huggingface.co/settings/tokens

```python
# 输入账号topken
from huggingface_hub import notebook_login
notebook_login()

# 记录我们要传的模型以及用到的超参
model = {
    "env_id": env_id,
    "max_steps": max_steps,
    "n_training_episodes": n_training_episodes,
    "n_eval_episodes": n_eval_episodes,
    "eval_seed": eval_seed,

    "learning_rate": learning_rate,
    "gamma": gamma,

    "max_epsilon": max_epsilon,
    "min_epsilon": min_epsilon,
    "decay_rate": decay_rate,

    "qtable": Qtable_frozenlake
}

# push
username = "TUMxudashuai" # FILL THIS
repo_name = "q-FrozenLake-v1-4x4-noSlippery"
push_to_hub(
    repo_id=f"{username}/{repo_name}",
    model=model,
    env=env)
```

### 2.2 Taxi-v3

#### 2.2.1 安装环境

同2.1.1

#### 2.2.2 导入包

同2.1.2

#### 2.2.3 建立环境

- 环境信息：https://www.gymlibrary.dev/environments/toy_text/taxi/

```
env = gym.make("Taxi-v3")
```

#### 2.2.4 创建并初始化Q-table

```
state_space = env.observation_space.n
print("There are ", state_space, " possible states")
action_space = env.action_space.n
print("There are ", action_space, " possible actions")

Qtable_taxi = initialize_q_table(state_space, action_space)
print(Qtable_taxi)
print("Q-table shape: ", Qtable_taxi .shape)
```



#### 2.2.5 定义epsilon-greedy policy

同2.1.5

#### 2.2.6 定义greedy policy

同2.1.6

#### 2.2.7 定义hyperparameters

```python
# Training parameters
n_training_episodes = 25000   # Total training episodes
learning_rate = 0.7           # Learning rate

# Evaluation parameters
n_eval_episodes = 100        # Total number of test episodes

# DO NOT MODIFY EVAL_SEED
eval_seed = [16,54,165,177,191,191,120,80,149,178,48,38,6,125,174,73,50,172,100,148,146,6,25,40,68,148,49,167,9,97,164,176,61,7,54,55,
 161,131,184,51,170,12,120,113,95,126,51,98,36,135,54,82,45,95,89,59,95,124,9,113,58,85,51,134,121,169,105,21,30,11,50,65,12,43,82,145,152,97,106,55,31,85,38,
 112,102,168,123,97,21,83,158,26,80,63,5,81,32,11,28,148] # Evaluation seed, this ensures that all classmates agents are trained on the same taxi starting position
                                                          # Each seed has a specific starting state

# Environment parameters
env_id = "Taxi-v3"           # Name of the environment
max_steps = 99               # Max steps per episode
gamma = 0.95                 # Discounting rate

# Exploration parameters
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.05           # Minimum exploration probability 
decay_rate = 0.005            # Exponential decay rate for exploration prob
```

#### 2.2.8 建立训练循环

同2.1.8

#### 2.2.9 训练模型

```python
Qtable_taxi = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable_taxi)
```



#### 2.2.10 发布模型到Huggingface

前置操作同2.1.10

登录后：

```python
# 记录自己的模型(q-table)和用到的超参
model = {
    "env_id": env_id,
    "max_steps": max_steps,
    "n_training_episodes": n_training_episodes,
    "n_eval_episodes": n_eval_episodes,
    "eval_seed": eval_seed,

    "learning_rate": learning_rate,
    "gamma": gamma,

    "max_epsilon": max_epsilon,
    "min_epsilon": min_epsilon,
    "decay_rate": decay_rate,

    "qtable": Qtable_taxi
}

# 上传到自己的库
username = "TUMxudashuai" # FILL THIS
repo_name = "q-Taxi-v3"
push_to_hub(
    repo_id=f"{username}/{repo_name}",
    model=model,
    env=env)
```

#### 2.2.11 从Huggingface上下载模型

```python
from urllib.error import HTTPError

from huggingface_hub import hf_hub_download

#---------------- 下载函数定义 ----------------
def load_from_hub(repo_id: str, filename: str) -> str:
    """
    Download a model from Hugging Face Hub.
    :param repo_id: id of the model repository from the Hugging Face Hub
    :param filename: name of the model zip file from the repository
    """
    try:
        from huggingface_hub import cached_download, hf_hub_url
    except ImportError:
        raise ImportError(
            "You need to install huggingface_hub to use `load_from_hub`. "
            "See https://pypi.org/project/huggingface-hub/ for installation."
        )

    # Get the model from the Hub, download and cache the model on your local disk
    pickle_model = hf_hub_download(
        repo_id=repo_id,
        filename=filename
    )

    with open(pickle_model, 'rb') as f:
      downloaded_model_file = pickle.load(f)
    
    return downloaded_model_file

#---------------- 下载taxi-v3模型，并使用----------------
model = load_from_hub(repo_id="TUMxudashuai/q-Taxi-v3", filename="q-learning.pkl")
print(model)
env = gym.make(model["env_id"])
evaluate_agent(env, model["max_steps"], model["n_eval_episodes"], model["qtable"], model["eval_seed"])

#---------------- 下载Frozenlake模型，并使用 ----------------
model = load_from_hub(repo_id="TUMxudashuai/q-FrozenLake-v1-4x4-noSlippery", filename="q-learning.pkl")
env = gym.make(model["env_id"], is_slippery=False)
evaluate_agent(env, model["max_steps"], model["n_eval_episodes"], model["qtable"], model["eval_seed"])
```



## 3. Deep Q-Learning

https://colab.research.google.com/github/huggingface/deep-rl-class/blob/main/unit3/unit3.ipynb

### 3.1 Atari' Space Invaders

#### 3.1.1 setup a virtual display

```python
%%capture
pip install pyglet==1.5.1 
sudo apt install python-opengl
sudo apt install ffmpeg
sudo apt install xvfb
pip3 install pyvirtualdisplay

# Additional dependencies for RL Baselines3 Zoo
apt-get install swig cmake freeglut3-dev 

# Virtual display
from pyvirtualdisplay import Display

virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()
```



#### 3.1.2 下载[RL-Baseline3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo)

RL Baselines3 Zoo: A Training Framework for Stable Baselines3 Reinforcement Learning Agents

```python
!git clone https://github.com/DLR-RM/rl-baselines3-zoo
%cd /content/rl-baselines3-zoo/
!pip install -r requirements.txt
!pip install huggingface_sb3
```

- 如果遇到AutoROM的问题，导致无法下载stable-baseline3[extra]

  - 第一步：

    ```
    pip install autorom
    AutoROM --install-dir /path/to/install	#安装地址
    AutoROM --accept-license
    ```

  - 第二步：

    ```
    pip install sb3-contrib
    pip install optuna
    ```

  - 第三步：

    修改/home/xuy1fe/.conda/envs/RL-learning/lib/python3.8/site-packages/stable_baselines3/common/atari_wrappers.py的第36行

    ```
    原来：noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
    改为：noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
    ```

#### 3.1.3 设置hyperparameters

使用RL-Baseline3 Zoo框架做RL时，在[rl-baselines3-zoo/hyperparams/dqn.yml](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml)处定义超参。

```yml
atari:
  env_wrapper:
    - stable_baselines3.common.atari_wrappers.AtariWrapper
  frame_stack: 4
  policy: 'CnnPolicy'
  n_timesteps: !!float 1e6
  buffer_size: 100000
  learning_rate: !!float 1e-4
  batch_size: 32
  learning_starts: 100000
  target_update_interval: 1000
  train_freq: 4
  gradient_steps: 1
  exploration_fraction: 0.1
  exploration_final_eps: 0.01
  # If True, you need to deactivate handle_timeout_termination
  # in the replay_buffer_kwargs
  optimize_memory_usage: False
```

- 各个参数的意义见[stable-baseline3](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html#parameters) 
  - 我们将4个frame stack起来
  - 我们使用CnnPolicy来训练
  - 训练1million steps

#### 3.1.4 train

使用RL-Baseline3 Zoo框架做RL时，使用[rl-baselines3-zoo/train.py](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/train.py)来训练

```
python train.py --algo dqn  --env SpaceInvadersNoFrameskip-v4 -f logs/
```

#### 3.1.5 Evaluate

使用RL-Baseline3 Zoo框架做RL时，使用[rl-baselines3-zoo/enjoy.py](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/enjoy.py)来评估

```
python enjoy.py  --algo dqn  --env SpaceInvadersNoFrameskip-v4  --no-render  --n-timesteps 5000  --folder logs/
```

#### 3.1.6 上传到Huggingface

Token: https://huggingface.co/settings/tokens

1. 登陆账号

```shell
# 如果直接用终端
huggingface-cli login

# 如果用jupyter
from huggingface_hub import notebook_login # To log to our Hugging Face account to be able to upload models to the Hub.
notebook_login()
!git config --global credential.helper store
```

2. 上传

```
python -m rl_zoo3.push_to_hub  --algo dqn  --env SpaceInvadersNoFrameskip-v4  --repo-name dqn-SpaceInvadersNoFrameskip-v4  -orga TUMxudashuai  -f logs/
```



#### 3.1.7 从Huggingface下载模型

```shell
# Download model and save it into the logs/ folder
python -m rl_zoo3.load_from_hub --algo dqn --env BeamRiderNoFrameskip-v4 -orga sb3 -f rl_trained/
# 评估下载的模型
python enjoy.py --algo dqn --env BeamRiderNoFrameskip-v4 -n 5000  -f rl_trained/
```



## 4. Unity MLAgents

https://colab.research.google.com/github/huggingface/deep-rl-class/blob/main/unit4/unit4.ipynb

cuda版本有问题，未找到如何在本地跑的方法

### 4.1 下载ml-agents

```shell
git clone https://github.com/huggingface/ml-agents/
cd ml-agents
pip3 install -e ./ml-agents-envs
pip3 install -e ./ml-agents
```

### 4.2 下载训练环境pyramids

```shell
# 存放环境的地址
mkdir ./trained-envs-executables/linux -p
# 用wget从google drive：https://drive.google.com/uc?export=download&id=1UiFNdKlsH0NTu32xV-giYUEVKV4-vc7H 下载环境
# 也可以直接下载然后放到上面那个文件夹
!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1UiFNdKlsH0NTu32xV-giYUEVKV4-vc7H' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1UiFNdKlsH0NTu32xV-giYUEVKV4-vc7H" -O ./trained-envs-executables/linux/Pyramids.zip && rm -rf /tmp/cookies.txt
# 解压下载的文件
unzip -d ./trained-envs-executables/linux/ ./trained-envs-executables/linux/Pyramids.zip
# 给文件权限
chmod -R 755 ./trained-envs-executables/linux/Pyramids/Pyramids
```

### 4.3 修改hyerparameters

ML-Agents训练用的超参，保存在ml-agents/config/的对应yaml文件中。ML-Agent各个超参的定义见[官网](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Training-Configuration-File.md)

这里我们修改ml-agents/config/ppo/PyramidsRND.yaml文件

```
max_steps: 3000000
改成
max_steps: 500000
```

### 4.4 训练agent

设置4个参数：

1. `mlagents-learn <config>`: the path where the hyperparameter config file is.
2. `--env`: where the environment executable is.
3. `--run_id`: the name you want to give to your training run id.
4. `--no-graphics`: to not launch the visualization during the training.

```shell
mlagents-learn ./config/ppo/PyramidsRND.yaml --env=./trained-envs-executables/linux/Pyramids/Pyramids --run-id="Pyramids Training" --no-graphics
```



### 4.5 上传到hugging face

- 登陆

    ```shell
    # if no jupyter:huggingface-cli login
    from huggingface_hub import notebook_login
    notebook_login()
    ```
    
- 上传

  使用`mlagents-push-to-hf`上传，设置如下参数：
  
  - `--run-id`: the name of the training run id.
  - `--local-dir`: where the agent was saved, it’s results/, so in my case results/First Training.
  - `--repo-id`: the name of the Hugging Face repo you want to create or update. It’s always / If the repo does not exist **it will be created automatically**
  - `--commit-message`: since HF repos are git repository you need to define a commit message.
  
  ```shell
  !mlagents-push-to-hf --run-id="Pyramids Training" --local-dir="./results/Pyramids Training" --repo-id="TUMxudashuai/testpyramidsrnd" --commit-message="First Pyramids"
  ```
  
  
### 4.6 Visual agent online

https://huggingface.co/spaces/unity/ML-Agents-Pyramids

在上述网址，选择4.5中上传的模型



## 5. Policy Gradient with PyTorch

https://colab.research.google.com/github/huggingface/deep-rl-class/blob/main/unit5/unit5.ipynb



# 副1、 Optuna: hyperparameter optimization

[Optuna](https://github.com/optuna/optuna) is an automatic hyperparameter optimization software framework, particularly designed for machine learning.

## 0. 自动超参搜索 资料汇总

1. [深度学习模型超参数搜索实用指南](https://zhuanlan.zhihu.com/p/45353509)

   主要有4种搜索策略

   1. **Babysitting (或Grad student Descent)**人工搜索
   2. **网格搜索 Grid Search**
   3. **随机搜索 Random Search**
   4. **贝叶斯优化 Bayesian Optimization**

## 1. 下载依赖

```shell
!pip install stable-baselines3
# Optional: install SB3 contrib to have access to additional algorithms
!pip install sb3-contrib
# Optuna will be used in the last part when doing hyperparameter tuning
!pip install optuna
```

## 2. 导入库和算法

```python
import gym
import numpy as np
from stable_baselines3 import PPO, A2C, SAC, TD3, DQN
# Algorithms from the contrib repo
# https://github.com/Stable-Baselines-Team/stable-baselines3-contrib
from sb3_contrib import QRDQN, TQC
from stable_baselines3.common.env_util import make_vec_env	# Parallel environments
from stable_baselines3.common.evaluation import evaluate_policy
```

## 3. The Importance of tuned Hyperparameters

- When compared with Supervised Learning, Deep Reinforcement Learning is far more sensitive to the choice of hyper-parameters such as learning rate, number of neurons, number of layers, optimizer ... etc.

- 除了超参，算法的选择也是很重要的。

- 下面是用不同算法的选择 和 调整参数 对学习[Pendulum](https://www.gymlibrary.dev/environments/classic_control/pendulum/)的影响比较：

  首先导入环境

  ```python
  env_id = "Pendulum-v1"
  # Env used only for evaluation
  eval_envs = make_vec_env(env_id, n_envs=10)
  # 4000 training timesteps
  budget_pendulum = 4000
  ```

### 3.1 PPO

1. 训练4000步(20 episodes)

    ```python
    ppo_model = PPO("MlpPolicy", env_id, seed=0, verbose=0).learn(budget_pendulum)
    mean_reward, std_reward = evaluate_policy(ppo_model, eval_envs, n_eval_episodes=100, deterministic=True)
    
    print(f"PPO Mean episode reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    ```

    得到：PPO Mean episode reward: -1146.01 +/- 253.07

    可见效果不好，尝试步骤2：

2. 训练的更长40000步

    ```python
    new_budget = 10 * budget_pendulum
    ppo_model = PPO("MlpPolicy", env_id, seed=0, verbose=0).learn(new_budget)
    mean_reward, std_reward = evaluate_policy(ppo_model, eval_envs, n_eval_episodes=100, deterministic=True)
    
    print(f"PPO Mean episode reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    ```

    得到：PPO Mean episode reward: -1164.48 +/- 205.87

    可见效果一般，尝试步骤3：

3.  tune Hyperparameters

    ```python
    tuned_params = {
        "gamma": 0.9,
        "use_sde": True,
        "sde_sample_freq": 4,
        "learning_rate": 1e-3,
    }
    
    # budget = 10 * budget_pendulum
    ppo_tuned_model = PPO("MlpPolicy", env_id, seed=1, verbose=1, **tuned_params).learn(50_000, log_interval=5)
    
    mean_reward, std_reward = evaluate_policy(ppo_tuned_model, eval_envs, n_eval_episodes=100, deterministic=True)
    
    print(f"Tuned PPO Mean episode reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    ```

    得到：Tuned PPO Mean episode reward: -173.78 +/- 99.40

    可见效果好了很多

### 3.2 [A2C](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html)

```python
a2c_model = A2C("MlpPolicy",env_id,verbose=1)
mean_reward, std_reward = evaluate_policy(a2c_model, eval_envs, n_eval_episodes=100, deterministic=True)

print(f"A2C Mean episode reward: {mean_reward:.2f} +/- {std_reward:.2f}")
```

得到：A2C Mean episode reward: -1210.47 +/- 331.78



## 4. Grad Student Descent

这种方法是**100％手动**，是研究人员，学生和业余爱好者最广泛采用的方法。

- 用A2C算法学习CartPole-v1任务：

  ```python
  budget = 20_000
  eval_envs_cartpole = make_vec_env("CartPole-v1", n_envs=10)
  model = A2C("MlpPolicy", "CartPole-v1", seed=8, verbose=1).learn(budget)
  mean_reward, std_reward = evaluate_policy(model, eval_envs_cartpole, n_eval_episodes=50, deterministic=True)
  
  print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
  ```

  得到mean_reward:140.40 +/- 54.13

  然后我们尝试手动调参，打败上面这个分数，最优解得分是500：

- Grad Student Descent

  ```python
  import torch.nn as nn
  policy_kwargs = dict(
      net_arch=[
        dict(vf=[64, 64], pi=[64, 64]), # network architectures for actor/critic
      ],
      activation_fn=nn.Tanh,
  )
  
  hyperparams = dict(
      n_steps=5, # number of steps to collect data before updating policy
      learning_rate=7e-4,
      gamma=0.99, # discount factor
      max_grad_norm=0.5, # The maximum value for the gradient clipping
      ent_coef=0.0, # Entropy coefficient for the loss calculation
  )
  
  model = A2C("MlpPolicy", "CartPole-v1", seed=8, verbose=1, **hyperparams).learn(budget)
  
  mean_reward, std_reward = evaluate_policy(model, eval_envs_cartpole, n_eval_episodes=50, deterministic=True)
  
  print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
  ```

  得到mean_reward:132.30 +/- 34.88

  显然手动调很麻烦

## 5. Automatic Hyperparameter Tuning

 Create a script that allows to search for the best hyperparameters automatically.

### 5.1 导入optuna库

```python
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances
```

### 5.2 Config

设置optuna的参数

```python
N_TRIALS = 100  # Maximum number of trials
N_JOBS = 1 # Number of jobs to run in parallel
N_STARTUP_TRIALS = 5  # Stop random sampling after N_STARTUP_TRIALS
N_EVALUATIONS = 2  # Number of evaluations during the training
N_TIMESTEPS = int(2e4)  # Training budget
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_ENVS = 5
N_EVAL_EPISODES = 10
TIMEOUT = int(60 * 15)  # 15 minutes

ENV_ID = "CartPole-v1"

DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy",
    "env": ENV_ID,
}
```

### 5.3 Define search space

定义optuna想要找那些超参，以及他们的范围

```python
from typing import Any, Dict
import torch
import torch.nn as nn
# 变量名后面的冒号是：类型注解，3.6以后加入的，冒号右边是类型，仅仅是注释，有些鸡肋
def sample_a2c_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for A2C hyperparameters.

    :param trial: Optuna trial object
    :return: The sampled hyperparameters for the given trial.
    """
    # Discount factor between 0.9 and 0.9999
    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)
    # 8, 16, 32, ... 1024
    n_steps = 2 ** trial.suggest_int("exponent_n_steps", 3, 10)

    ### YOUR CODE HERE
    # TODO:
    # - define the learning rate search space [1e-5, 1] (log) -> `suggest_float`
    # - define the network architecture search space ["tiny", "small"] -> `suggest_categorical`
    # - define the activation function search space ["tanh", "relu"]
    learning_rate =trial.suggest_float("lr",1e-5, 1, log=True)
    net_arch = trial.suggest_categorical("net_arch",["tiny","small"])
    activation_fn = trial.suggest_categorical("activation_fn",["tanh","relu"])

    ### END OF YOUR CODE

    # Display true values
    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("n_steps", n_steps)

    net_arch = [
        {"pi": [64], "vf": [64]}
        if net_arch == "tiny"
        else {"pi": [64, 64], "vf": [64, 64]}
    ]

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]

    return {
        "n_steps": n_steps,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "max_grad_norm": max_grad_norm,
        "policy_kwargs": {
            "net_arch": net_arch,
            "activation_fn": activation_fn,
        },
    }
```

### 5.4 Define the Callback function

define a custom callback to report the results of periodic evaluations to Optuna:

```python
from stable_baselines3.common.callbacks import EvalCallback

class TrialEvalCallback(EvalCallback):
    """
    Callback used for evaluating and reporting a trial.
    
    :param eval_env: Evaluation environement
    :param trial: Optuna trial object
    :param n_eval_episodes: Number of evaluation episodes
    :param eval_freq:   Evaluate the agent every ``eval_freq`` call of the callback.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic policy.
    :param verbose:
    """

    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):
		# super() 函数是用于调用父类(超类)的一个方法。
		# super() 是用来解决多重继承问题的，直接用类名调用父类方法在使用单继承的时候没问题，但是如果使用多继承，会涉及到查找顺序（MRO）、重复调用（钻石继承）等种种问题。
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Evaluate policy (done in the parent class)
            # 评价体系在stable_baselines3.common.callbacks的EvalCallback中已定义，要找mean reward最高的
            # 也是5.6中objective value的值
            super()._on_step()
            self.eval_idx += 1
            # Send report to Optuna
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # 根据trial的pruning algorithm来判断是否需要停止当前参数的训练
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True
```

### 5.5 Define the objective function

define the objective function that is in charge of sampling hyperparameters, creating the model and then returning the result to Optuna

```python
def objective(trial: optuna.Trial) -> float:
    """
    Objective function using by Optuna to evaluate
    one configuration (i.e., one set of hyperparameters).

    Given a trial object, it will sample hyperparameters,
    evaluate it and report the result (mean episodic reward after training)

    :param trial: Optuna trial object
    :return: Mean episodic reward after training
    """

    kwargs = DEFAULT_HYPERPARAMS.copy()
    ### YOUR CODE HERE
    # TODO: 
    # 1. Sample hyperparameters and update the keyword arguments
    # 用5.3的取样函数，在搜索范围内排列组合参数，并返回
    # 使用字典的update函数，将返回值更新到kwargs中去
    kwargs.update(sample_a2c_params(trial))
    # 创建RL model
    model = A2C(**kwargs)

    # 2. Create envs used for evaluation using `make_vec_env`, `ENV_ID` and `N_EVAL_ENVS`
    eval_envs = make_vec_env(ENV_ID,N_EVAL_ENVS)

    # 3. Create the `TrialEvalCallback` callback defined above that will periodically evaluate
    # and report the performance using `N_EVAL_EPISODES` every `EVAL_FREQ`
    # TrialEvalCallback signature:
    # TrialEvalCallback(eval_env, trial, n_eval_episodes, eval_freq, deterministic, verbose)
    # 使用5.4的回调函数，返回每个参数组合的效果
    eval_callback = TrialEvalCallback(eval_envs,
                                      trial, 
                                      N_EVAL_EPISODES, 
                                      EVAL_FREQ,
                                      deterministic=True)

    ### END OF YOUR CODE

    nan_encountered = False
    try:
        # Train the model
        model.learn(N_TIMESTEPS, callback=eval_callback)
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN
        print(e)
        nan_encountered = True
    finally:
        # Free memory
        model.env.close()
        eval_envs.close()

    # Tell the optimizer that the trial failed
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward
```

### 5.6 optimization loop

调用5.3，5.4，5.5的代码，不断循环找出最优的参数设置

```python
import torch as th

# Set pytorch num threads to 1 for faster training
th.set_num_threads(1)
# Select the sampler, can be random, TPESampler, CMAES, ...
sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
# Do not prune before 1/3 of the max budget is used
# 在1/3 budget预算后，会prune the least promising trials
# 每个trials代表一组参数设置组合
pruner = MedianPruner(
    n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3
)
# Create the study and start the hyperparameter optimization
study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")

try:
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS, timeout=TIMEOUT)
except KeyboardInterrupt:
    pass

print("Number of finished trials: ", len(study.trials))

print("Best trial:")
trial = study.best_trial

print(f"  Value: {trial.value}")

print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

print("  User attrs:")
for key, value in trial.user_attrs.items():
    print(f"    {key}: {value}")

# Write report
study.trials_dataframe().to_csv("study_results_a2c_cartpole.csv")

fig1 = plot_optimization_history(study)
fig2 = plot_param_importances(study)

fig1.show()
fig2.show()
```

最后得到

```python
Number of finished trials:  60
Best trial:
  Value: 500.0
  Params: 
    gamma: 0.007193710709881038
    max_grad_norm: 1.971438140124285
    exponent_n_steps: 8
    lr: 0.007340303426309211
    net_arch: tiny
    activation_fn: relu
  User attrs:
    gamma_: 0.9928062892901189
    n_steps: 256
```



# 副2、 Curiosity-Driven Learning

[Curiosity-Driven Learning through Next State Prediction](https://medium.com/data-from-the-trenches/curiosity-driven-learning-through-next-state-prediction-f7f4e2f592fa)

[Random Network Distillation: a new take on Curiosity-Driven Learning](https://medium.com/data-from-the-trenches/curiosity-driven-learning-through-random-network-distillation-488ffd8e5938)

## 1. Two Major Problems in Modern RL

### 1.1 [sparse rewards](https://zhuanlan.zhihu.com/p/558034131)稀疏奖励  problem

对应于dense rewards稠密奖励

- most rewards do not contain information, and hence are set to zero.完成目标事件的次数太少或者完成目标的步数太长，导致奖励空间中的负奖励样本远远多于正奖励样本数。

- 问题的原因：

  RL是基于reward hypothesis的算法，也就是说RL的目标是最大化cumulative rewards。因此如果Rewards一直都是0，agent也就不知道他们的action是否appropriate恰当，最后agent进步缓慢。

- 例子：

  1. 在围棋中，从开始下棋到棋局结束才能判断胜负，此时智能体才能获得奖励，棋局中间过程中的奖励很难评价；
  2. 在导航任务中，智能体只有在规定的时间步内到达指定位置才能得到奖励，中间过程的每一步都是无奖励的；
  3. 在机械臂抓取任务中，机械臂通过完成一系列复杂的位姿控制成功抓取目标后才能获得奖励，中间任何一步的失败都导致无法获得奖励。

  在这些任务过程中，如果agent没有得到有效的反馈(dense rewards), 它会花费大量时间学习最优策略并且到处乱试却找不到目标。

### 1.2  extrinsic reward function is handmade手工的

- 在每一个环境中，人们必须手动Implement实施 reward function。但我们如何scale度量一个大而复杂的环境呢？

## 2. What is Curiosity

- 为了解决上面这两个问题，我们需要开发一个reward function，它对于agent是intrinsic内在的(generated by the agent itself). The agent will act as a self-learner since it will be the student, but also its own feedback master.

  这个intrinsic reward mechanism机制，也被称为 **curiosity**, 因为它的reward会不断让agent去explore novel/unfamiliar states.

   In order to achieve that, our agent will receive a high reward when exploring new trajectories.

## 3. 两种计算curiosity的方法

### 3.1 Curiosity-Driven Learning through Next State Prediction

[Curiosity-Driven Learning through Next State Prediction](https://medium.com/data-from-the-trenches/curiosity-driven-learning-through-next-state-prediction-f7f4e2f592fa)是一个经典方法。
$$
IR=||predicted(s_{t+1})-s_{t+1} ||
$$

- Intrinsic reward(IR): prediction error in predicting $s_{t+1}$ given $s_t$ and $a_t$
- 我们会得到
  - small IR in familiar states: 因为 easy to predict next state
  - big IR in unfamiliar states: 因为 hard to predict next state in unknown trajectories
- Using curiosity will push our agent to favor transitions with high prediction error (which will be higher in areas where the agent has spent less time, or in areas with complex dynamics) **and consequently better explore our environment.**

### 3.2 Random Network Distillation

ML-Agents使用了一个更先进的方法：[Random Network Distillation: a new take on Curiosity-Driven Learning](https://medium.com/data-from-the-trenches/curiosity-driven-learning-through-random-network-distillation-488ffd8e5938)



