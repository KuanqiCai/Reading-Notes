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

### 3.1 Policy Gradient公式推导

将下面抽象的公式1 推导出可以用代码写出的近似公式。

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
         \nabla_\theta J(\pi_\theta) &= \mathop{E}\limits_{\tau\sim\pi_\theta}\big[\sum_{t=0}^T\nabla_\theta log\pi_\theta(a_t|s_t)R(\tau) \big]	\tag{6}
     \end{align}
     $$
     

- 由上面6式可知，策略梯度$\nabla_\theta J(\pi_\theta)$可以表示为一个期望，即我们可以用sample mean样本均值来近似他，由此思路得到公式：
  $$
  \hat{g}=\frac{1}{|D|}\sum_{\tau\in D}\sum_{t=0}^T\nabla_\theta log\pi_\theta(a_t|s_t)R(\tau) \tag{7：可用于编程计算的最简表达式}
  $$

  - $D={\{\tau_i\}}_{i=1\cdots N}$:所有的轨迹
  - $|D|$:轨迹的个数，=N

### 3.2 代码实现3.1的7式

```python
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box

# Multilayer Perceptron多层感知器，是一种前向结构的人工神经网路，映射一组输入向量到一组输出向量
def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    # 中括号是列表
    layers = []
    for j in range(len(sizes)-1):
        # out = a if b else c: 如果b对out为a, 如果b不对out为c。
        act = activation if j < len(sizes)-2 else output_activation
        # nn.Linear（）用于设置网络中的全连接层
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    # nn.Sequential(*args)和nn.ModuleList(modules=None)都是pytorch提供的神经网络容器。
    # nn.Sequential(*layers)将列表通过非关键字形式传入该容器。
    # *args是可变参数，传入的参数个数可变，类型是tuple。作用是可以一次给函数传很多的参数
    # **kw是关键字参数，关键字参数允许你传入0个或任意个含不同参数名的参数，这些关键字参数在函数内部自动组装为一个dict
    return nn.Sequential(*layers)

def train(env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2, 
          epochs=50, batch_size=5000, render=False):

    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # make core of policy network
    logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])

    # make function to compute action distribution
    # categorical()方法可以产生概率分布。参数可以选logits或者probs，两种不同的概率值
    # 之后使用sample()采样，得到的是输入tensor的index 
    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    # make action selection function (outputs int actions, sampled from policy)
    # .item()方法，返回张量元素的值，注意：张量中只有一个元素才能调用该方法
    def get_action(obs):
        return get_policy(obs).sample().item()

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    # make optimizer
    optimizer = Adam(logits_net.parameters(), lr=lr)

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, done, _ = env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                                  )
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr)
```




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