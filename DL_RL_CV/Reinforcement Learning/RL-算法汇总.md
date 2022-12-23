内容基本来自[OpenAI](https://spinningup.openai.com/en/latest/index.html)

# Kinds of RL Algorithms

- 各种RL相关词汇见[[RL_Course_Notes#1.8 基本概念汇总 | 基本概念汇总]]

  ![](https://raw.githubusercontent.com/Fernweh-yang/Reading-Notes/cb388cd25dedf69ffadf05ae3a721bdc8d96694b/%E7%AC%94%E8%AE%B0%E9%85%8D%E5%A5%97%E5%9B%BE%E7%89%87/Reinforcement%20Learning/rl_algorithms_classification.svg)

## 1. Model-Free vs Model-Based RL

- Model:指的是一个function可以predict环境的state taransitions and rewards
- Model优点：
  - Model可以让agent **提前思考并作出规划**。
  - Agent然后可以将提前规划的结果提炼distill成学习的策略
- Model缺点：
  - 通常很难得到**a ground-truth model of the environment**
  - 此外，The bias in the model can be exploited by the agent, resulting in an agent which performs well with respect to the learned model, but behaves sub-optimally (or super terribly) in the real environment.
  - 基于模型的学习fundamentally本质上是很难的，即使付出了大量努力，可能还是没有回报。

## 2. Policy Optimization vs Q-Learning

Q-Learning和Deep Q-Learning都是value based方法。他们需要estimate估计一个value function Q作为intermediate step然后再去寻找最优的policy。而policy-based方法，就可以直接optimize policy。

### 2.1 Policy Optimization

- 这种方法通过如下两种方法优化policy $\pi_{\theta}(a|s)$的参数$\theta$

  1. 通过gradient ascent on the performance objective$J(\pi_{\theta})$

  2. 通过最大化local approximations of $J(\pi_{\theta})$

  这两种优化都遵循 **on-policy**: 每一次的update都只取决于最新policy下动作所产生的数据。
  
- 使用这种方法的例子：

  - [A2C / A3C](https://arxiv.org/abs/1602.01783), which performs gradient ascent to directly maximize performance,
  - and [PPO](https://arxiv.org/abs/1707.06347), whose updates indirectly maximize performance, by instead maximizing a *surrogate objective* function which gives a conservative保守的 estimate for how much $J(\pi_{\theta})$ will change as a result of the update.


### 2.2 Q-Learning

- 这种方法优化一个近似体$Q_\theta(s,a)$来近似optimal action-value function $Q^*(s,a)$。

  这种优化方式通常采用 **off-policy**: 每一次更新都使用训练中收集到的任意数据。

- 使用这种方法的例子

  - [DQN](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), a classic which substantially launched the field of deep RL,
  - and [C51](https://arxiv.org/abs/1707.06887), a variant that learns a distribution over return whose expectation is $Q^*$

### 2.3 Trade-offs between them

- Policy Optimization的优势在于它直接优化我们想要的(policy)，因此更加的stable和reliable

- Q-Learning需要通过训练$Q_\theta$(每个动作的概率Q值)来得到我们想要的，所以不太稳定。

  但Q-Learning可以更充分的使用收集到的数据，因为每次更新他们都基于所有的数据。

### 2.4 Interpolating between them

Policy Optimization和Q -Learning并不是不相容的，因此有些方法兼顾了他们两个：

- [DDPG](https://arxiv.org/abs/1509.02971), an algorithm which concurrently learns a deterministic policy and a Q-function by using each to improve the other,
- [SAC](https://arxiv.org/abs/1801.01290), a variant which uses stochastic policies, entropy regularization, and a few other tricks to stabilize learning and score higher than DDPG on standard benchmarks.

## 3. Policy Optimization基础

### 3.1 得到最简单的Policy Gradient

- 问题背景:

  假如我们有策略$\pi_\theta$，并想最大化我们的回报$J(\pi_\theta)=\mathop{E}\limits_{\tau\sim\pi_\theta}[R(\tau)]=\displaystyle \int_\tau P(\tau|\theta)R(\tau)$

  我们通过gradient ascent来优化我们policy:
  $$
  \theta_{k+1}=\theta_k+\alpha\nabla_\theta J(\pi_\theta)|_{\theta_k} \tag{1}
  $$
  $\nabla_\theta J(\pi_\theta)$: policy gradient
  
- 为了真正的能使用1式，我们需要将policy gradient转化成能数值计算的表达

  1. **Probability of a Trajectory**
      轨迹的定义见[[RL_Course_Notes#1.8.5 trajectories]]
    transition matrix: $P(s_{t+1}|s_t,a_t)$和policy: $\pi_\theta(a_t|s_t)$都可能是stochastic的, 在已知这两者概率的情况下，可以给出形成任意一条trajectory的概率
  $$
  P(\tau|\theta)=\rho_0(s_0)\prod_{t=0}^TP(s_{t+1}|s_t,a_t)\pi_\theta(a_t|s_t) \tag{2}
  $$
  
  ​		强化学习的任务就可以理解为是最大化所有trajectory上能够获取到的reward总和的期望
  
  2. **The Log-Derivative Trick**
  
     这个技巧是： log x的导数是1/x
  
     将技巧用于2式子则得到：
     $$
     \nabla_\theta P(\tau|\theta)=P(\tau|\theta)\nabla_\theta log P(\tau|\theta) \tag{3}
     $$
  
  3. **Log-Probability of a Trajectory**
  
     将2式两边取对数得到:
     $$
     logP(\tau|\theta)=log\rho_0(s_0)+\sum_{t=0}^T\big( log\ P(s_{t+1}|s_t,a_t)+log\ \pi_\theta(a_t|s_t)\big) \tag{4}
     $$
  
  4. **Grad-Log-Prob of a Trajectory.**
  
     将4式子对$\theta$求导，前两项与$\theta$无关所以导数为0，得到:
     $$
     \nabla_\theta logP(\tau|\theta)=\sum_{t=0}^T\nabla_\theta log\pi_\theta(a_t|s_t) \tag{5}
     $$
  
  5. **整合**
  
     利用上面的这些式子计算policy gradient:$\nabla_\theta J(\pi_\theta)$
     $$
     \begin{align}
       \nabla_\theta J(\pi_\theta) &= \nabla_\theta \mathop{E}\limits_{\tau\sim\pi_\theta}[R(\tau)] \\
         &= \nabla_\theta \displaystyle \int_\tau P(\tau|\theta) R(\tau) \tag{展开expection}\\
         &= \displaystyle \int_\tau \nabla_\theta P(\tau|\theta) R(\tau) \\
         &= \displaystyle \int_\tau  P(\tau|\theta)\ \nabla_\theta\  log\  P(\tau|\theta)\ R(\tau) \tag{使用3式的log trick}\\
         &= \mathop{E}\limits_{\tau\sim\pi_\theta}[\nabla_\theta\  log\  P(\tau|\theta)\ R(\tau)] \tag{写回expectation形式}\\
         (参考5式)&求导可得：\\
         \nabla_\theta J(\pi_\theta) &= \mathop{E}\limits_{\tau\sim\pi_\theta}\big[\sum_{t=0}^T\nabla_\theta log\pi_\theta(a_t|s_t)R(\tau) \big]
     \end{align}
     $$
     




# DDPG

## 1. 资料汇总

- [OpenAI Spinning Guide for DDPG](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)
- [DDPG Paper](https://arxiv.org/abs/1509.02971)

## 2. 基本概念

- Deep Deterministic Policy Gradient (DDPG) is an algorithm which concurrently同时的 learns a Q-function and a policy.
- It uses off-policy data and the Bellman equation to learn the Q-function, and uses the Q-function to learn the policy.

# TD3

## 1.  资料汇总

- [OpenAI Spinning guide on TD3](https://spinningup.openai.com/en/latest/algorithms/td3.html)
- [Original paper](https://arxiv.org/pdf/1802.09477.pdf)
- [Original Implementation](https://github.com/sfujim/TD3)

## 2.  