# 0、资料汇总

- [Documentation](https://www.gymlibrary.dev/)
- [知乎](https://zhuanlan.zhihu.com/p/33553076?utm_id=0)

# 1、Basic Usage

## 1.1 [Core](https://www.gymlibrary.dev/api/core/)

![](https://www.gymlibrary.dev/_images/AE_loop.png)

一个简单的例子：

```python
import gym
env = gym.make("LunarLander-v2")	# 1.运行一个环境
env.reset()							# 2.重置环境

for _ in range(10000):
    env.step(env.action_space.sample())
    env.render() 					# 3.用于显示一个窗口来render环境
    
from gym.utils.env_checker import check_env
check_env(env,skip_render_check=False,warn=False) # 4.检查环境是否conform to Gym Api
												  # 后面两个参数都是默认true，是否检查render()和输出warn
env.close()							# 5.关闭环境
```

## 1.2 [Space](https://www.gymlibrary.dev/api/spaces/)

用于定义 action 和 observation space，他们都要inherit from Space类。

Gym有不同的Space类型：

- `Box`: describes an n-dimensional continuous space. It’s a bounded space where we can define the upper and lower limits which describe the valid values our observations can take.
- `Discrete`: describes a discrete space where {0, 1, …, n-1} are the possible values our observation or action can take. Values can be shifted to {a, a+1, …, a+n-1} using an optional argument.
- `Dict`: represents a dictionary of simple spaces.
- `Tuple`: represents a tuple of simple spaces.
- `MultiBinary`: creates a n-shape binary space. Argument n can be a number or a `list` of numbers.
- `MultiDiscrete`: consists of a series of `Discrete` action spaces with a different number of actions in each element.

```python
>>> from gym.spaces import Box, Discrete, Dict, Tuple, MultiBinary, MultiDiscrete
>>> import numpy as np
# Box:
>>> observation_space = Box(low=-1.0, high=2.0, shape=(3,), dtype=np.float32)
>>> observation_space.sample()
array([-0.30874878, -0.44607827,  1.8394998 ], dtype=float32)

# Discrete:
>>> observation_space = Discrete(5)
>>> observation_space.sample()
4

```



## 1.3 [Wrappers](https://www.gymlibrary.dev/api/wrappers/)

可以不用改变源代码，就让我们修改现有环境

```python
>>> import gym
>>> from gym.wrappers import RescaleAction

>>> base_env = gym.make("BipedalWalker-v3")						# 1.要修改的环境
>>> base_env.action_space										# 显示环境的动作空间
Box([-1. -1. -1. -1.], [1. 1. 1. 1.], (4,), float32)

>>> wrapped_env = RescaleAction(base_env, min_action=0, max_action=1)	# 2.修改环境
>>> wrapped_env.action_space
Box([0. 0. 0. 0.], [1. 1. 1. 1.], (4,), float32)

>>> wrapped_env.unwrapped										# 3.将修改的环境取消修改
```



## 1.4 Playing within environment

```python
from gym.utils.play import play			# 使用play函数
play(gym.make('Pong-v0'))
```

