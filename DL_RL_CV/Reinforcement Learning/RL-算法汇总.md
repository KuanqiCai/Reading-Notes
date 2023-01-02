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
  $\nabla_\theta J(\pi_\theta)$: **policy gradient**
  
- 为了真正的能使用1式，我们需要将policy gradient转化成能数值计算的表达

  1. **Probability of a Trajectory**
      轨迹的定义见[[RL_Course_Notes#1.8.5 trajectories]]
      transition matrix: $P(s_{t+1}|s_t,a_t)$和policy: $\pi_\theta(a_t|s_t)$都可能是stochastic的, 在已知这两者概率的情况下，可以给出形成任意一条trajectory的概率
  $$
  P(\tau|\theta)=\rho_0(s_0)\prod_{t=0}^TP(s_{t+1}|s_t,a_t)\pi_\theta(a_t|s_t) \tag{2}
  $$

  ​		强化学习的任务就可以理解为是最大化所有trajectory上能够获取到的reward总和的期望

  2. **The Log-Derivative Trick**

     这个技巧是： 
     $$
     \frac{d}{dx}(logf(x;y))=\frac{f'(x;y)}{f(x;y)}
     $$
     将技巧用于2式则得到：
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

- 参考1：[PyTorch distributions](https://pytorch.org/docs/stable/distributions.html) 基本过程如网站所诉：

  ```python
  # 由神经网络得到各个动作的概率
  probs = policy_network(state)
  # 得到动作的概率分布
  m = Categorical(probs)
  # 选择一个动作
  action = m.sample()
  # 执行动作得到新的state和reward
  next_state, reward = env.step(action)
  # 计算损失函数，公式见3.1的7式子后半段
  loss = -m.log_prob(action) * reward
  # 将损失loss反响传播，计算梯度
  loss.backward()
  ```

- 运行: `python policy_gradient.py`

- 实现的功能：

  - main()：创建CartPole-v0环境，在这个环境中调用train()训练
  - mlp()：创建一个神经网络，用于生成动作的概率
  - train()：根据3.1的7式，计算策略梯度，执行训练

- 48行categorical distribution使用的是logits，每一个输出(action)的概率由logits的softmax function给出：
  $$
  p_j=\frac{exp(x_j)}{\sum_i exp(x_i)}
  $$

  - 动作j在[logit](https://zhuanlan.zhihu.com/p/27188729) $x_j$下的概率，$x_j=logit(j)=log(\frac{j}{1-j})$
    - odds: $\frac{j}{1-j}$范围是$[0,\infin]$
    - logit: $log(\frac{j}{1-j})$范围是$[-\infin,\infin]$

```python
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box

# Multilayer Perceptron多层感知器，是一种前向结构的人工神经网路，映射一组输入向量到一组输出向量
# 作用：创建一个神经网络，用于生成动作的概率
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
    # 这里mlp返回的是一个nn.Sequential()构建的类模型。
    logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])

    # make function to compute action distribution
    # 作用：得到各个动作的概率分布
    # categorical()方法可以产生概率分布。参数可以选logits或者probs，两种不同的概率值
    # 之后使用sample()采样，得到的是输入tensor的index 
    # obs就是state
    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    # make action selection function (outputs int actions, sampled from policy)
    # 作用：得到某一状态下所选择的动作
    # .item()方法，返回张量元素的值，注意：张量中只有一个元素才能调用该方法
    def get_action(obs):
        return get_policy(obs).sample().item()

    # make loss function whose gradient, for the right data, is policy gradient
    # 作用：计算策略梯度policy gradient，策略梯度的定义见3.1的1式
    # right data：一个(state,action,weight)元组，由当前策略下行动后所收集得到的数据。
    # 这里state-action pair的权重weights是他们所属episode的回报return
    """
    这里虽然叫loss function，但不是传统意义上监督学习中的loss function,有2个区别：
    1. 通常损失函数定义于一个固定的数据分布，而这里数据由当前策略采样而得到，所以这里数据分布是根据Parameters变动的
    2. 通常损失函数会评估我们关心的性能指标，而这里我们虽然关心expected return，但并不会近似它
    """
    # 计算的是3.1中的7式
    def compute_loss(obs, act, weights):
        # 返回当前policy下act的概率的对数。
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    # make optimizer
    # 使用Adaptive Moment Estimation自适应矩估计，来作梯度优化计算
    optimizer = Adam(logits_net.parameters(), lr=lr)

    # for training policy
    # 作用：训练一个epoch的策略梯度
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
        ep_rews = []            # list for rewards accrued积累 throughout ep

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
        # zero_grad(): Sets the gradients of all optimized torch.Tensor s to zero.
        # 将所有的张量归零
        optimizer.zero_grad()
        # 计算损失
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                                  )
        # 反响传播，计算梯度
        batch_loss.backward()
        # 更新参数
        optimizer.step()
        return batch_loss, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

if __name__ == '__main__':
    import argparse	# 这个库让我们直接在命令行中就可以向程序中传入参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr)
```

### 3.3 Reward-to-go Policy gradient

[参考](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#don-t-let-the-past-distract-you)

- 3.1中的6式：
  $$
  \nabla_\theta J(\pi_\theta) = \mathop{E}\limits_{\tau\sim\pi_\theta}\big[\sum_{t=0}^T\nabla_\theta log\pi_\theta(a_t|s_t)R(\tau) \big]
  $$

  - 使用这个公式的话，每一步梯度改变每个动作的log-probabilities的时候，都会与获得的所有的奖励之和$R(\tau)$有关。
  - 但Agent应该只根据行动的结果来reinforce下一个动作，他们之前获得奖励和本次行动的好坏无关，即应该只算在行动之后获得的奖励

- 为此将上式改成Reward-to-go Policy gradient：
  $$
  \nabla_\theta J(\pi_\theta) = \mathop{E}\limits_{\tau\sim\pi_\theta}\big[\sum_{t=0}^T\nabla_\theta log\pi_\theta(a_t|s_t)\sum_{t'=t}^T R(s_{t'},a_{t'},s_{t'+1}) \big]	\tag{1}
  $$

  - 这里将整个轨迹的reward和改成本次动作之后的reward和

### 3.4 代码实现3.3的公式

相比于3.2变动只有两处：

20行和108行

```python
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)
"""
变动1：
将整个轨迹的reward和改成本次动作之后的reward和，即reward to go
"""
def reward_to_go(rews):
    n = len(rews)
    # numpy.zeros_like(array)返回一个和输入数组相同类型和形状的全是0的数组
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs

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
    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    # make action selection function (outputs int actions, sampled from policy)
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
        batch_weights = []      # for reward-to-go weighting in policy gradient
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

                # the weight for each logprob(a_t|s_t) is reward-to-go from t
                """
                变动2：
                将3.2中的120行batch_weights += [ep_ret] * ep_len改成如下。
                即只算当前动作之后的reward
                """
                batch_weights += list(reward_to_go(ep_rews))

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
    print('\nUsing reward-to-go formulation of policy gradient.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr)
```



### 3.5 Baselines in Policy gradient

#### 3.5.1 Expected Grad-Log-Prob (EGLP) lemma.

对于一个变量x的parameterized probability distribution参数化概率分布$p_\theta$，它的log梯度的期望为0:
$$
\mathop{E}_{x \sim P_\theta}[\nabla_\theta log P_\theta(x)]=0 \tag{1}
$$
证明如下：

1. 众所周知，概率和为1：
   $$
   \displaystyle \int_xP_\theta(x)=1 \tag{2}
   $$

2. 计算两边的梯度
   $$
   \nabla_\theta \displaystyle \int_xP_\theta(x)=\nabla_\theta 1 =0 \tag{3}
   $$

3. 使用3.1中3式提到的The Log-Derivative Trick得到：
   $$
   \begin{align}
   	0&=\nabla_\theta \displaystyle \int_xP_\theta(x)\\
   	&=\displaystyle \int_x \nabla_\theta P_\theta(x)\\
   	&=\displaystyle \int_x P_\theta(x) \nabla_\theta logP_\theta(x) \tag{使用trick}
   	
   	\\
   	概率的积分即&期望:
   	\\
   	0&=\mathop{E}_{x\sim P_\theta}[\nabla_\theta logP_\theta(x)]	\tag{4}
   \end{align}
   $$

#### 3.5.2 Baselines

由3.3的1式有：
$$
\nabla_\theta J(\pi_\theta) = \mathop{E}\limits_{\tau\sim\pi_\theta}\big[\sum_{t=0}^T\nabla_\theta log\pi_\theta(a_t|s_t)\sum_{t'=t}^T R(s_{t'},a_{t'},s_{t'+1}) \big]	\tag{5}
$$
4式显然有：
$$
\begin{align}
\mathop{E}_{x\sim P_\theta}[\nabla_\theta logP_\theta(x)]=\mathop{E}\limits_{\tau\sim\pi_\theta}\big [\nabla_\theta log\pi_\theta(a_t|s_t) \big]=0\\
\mathop{E}\limits_{\tau\sim\pi_\theta}\big [\nabla_\theta log\pi_\theta(a_t|s_t) \underbrace{b(s_t)}_{baseline}\big]=0\tag{添加一个baseline不影响}
\end{align}
$$
所以5式可化为:
$$
\nabla_\theta J(\pi_\theta) = \mathop{E}\limits_{\tau\sim\pi_\theta}\big[\sum_{t=0}^T\nabla_\theta log\pi_\theta(a_t|s_t)\big( \sum_{t'=t}^T R(s_{t'},a_{t'},s_{t'+1}) -b(s_t)\big)\big]	\tag{6}
$$
最常用的baseline是 [on-policy value function](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#value-functions) $V^\pi(s_t)$，可以帮助减少在计算策略梯度时sample estimate样本估计的variance方差。

实际中$V^\pi(s_t)$是无法被准确计算的，而是用一个神经网络来近似得到它$V_\phi(s_t)$，这个神经网络会随着Policy一起更新。
大多数policy optimization算法中学习[[RL_Course_Notes#1.8.8 Value function]]的方法是minimize a mean-squared-error objective：
$$
\phi_k=arg\ \mathop{min}_\phi \ \mathop{E}_{s_t,\hat{R}_t\sim \pi_k}\big[\big( V_\phi(s_t)-\hat{R}_t\big)^2 \big]
$$

### 3.6 其他形式的Policy Gradient

Policy Gradient可以写成通式：
$$
\nabla_\theta J(\pi_\theta) = \mathop{E}\limits_{\tau\sim\pi_\theta}\big[\sum_{t=0}^T\nabla_\theta log\pi_\theta(a_t|s_t)\phi_t\big]	
$$
其中$\phi_t$可写成

- $\phi_t=R(\tau)$, [[RL-算法汇总#3.1 Policy Gradient公式推导|3.1的6式]]

- $\phi_t= \sum_{t'=t}^T R(s_{t'},a_{t'},s_{t'+1}) \big]$, [[RL-算法汇总#3.3 Reward-to-go Policy gradient|3.3的1式]] 

- $\phi_t = \sum_{t'=t}^T R(s_{t'},a_{t'},s_{t'+1}) -b(s_t)$ , [[RL-算法汇总#3.5 Baselines in Policy gradient| 3.5的6式]]

- 也可以写成[On-Policy Action-Value Function](https://spinningup.openai.com/en/latest/spinningup/extra_pg_proof2.html) :$\phi_t=Q^{\pi\theta}(s_t,a_t)$

- 也可写成[[RL_Course_Notes#1.8.11 Advantage Functions|Advantage Function]]: $\phi_t=A^{\pi\theta}(s_t,a_t)$

  


# DDPG

## 1. 资料汇总

^a84aac

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