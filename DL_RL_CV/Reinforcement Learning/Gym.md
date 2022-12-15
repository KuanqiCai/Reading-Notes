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

- `Box`: describes an n-dimensional continuous space连续空间. It’s a bounded space where we can define the upper and lower limits which describe the valid values our observations can take.

  举例:

  `Box(low=0.0, high=255, shape=(210,160,3), dtype=np.uint8) `表示有210\*160\*3=100800个处于0到255的维度的空间

  `Box(-1.0, 1.0, (4,), float32)`表示4个joint(2个臂部2个膝盖)，他们的motor speed处于-1到1之间

- `Discrete`: describes a discrete space离散空间 where {0, 1, …, n-1} are the possible values our observation or action can take. Values can be shifted to {a, a+1, …, a+n-1} using an optional argument.

  举例：

  `Discrete(n=4)` 表示4个action上下左右。

- `Dict`: represents a dictionary of simple spaces.字典，用于整合一些空间实例比如Dict下带2个Box

  举例:

  ```python
  Dict({
  	'position':Discrete(2), 
  	'velocity':Discrete(3)
  })
  ```

- `Tuple`: represents a tuple of simple spaces.元组，用于整合一些空间实例比如Tuple下带2个Box。

  举例:

  ```python
  Tuple((
      Discrete(2), 
      Box(-1, 1, shape=(2,))
  ))
  ```

- `MultiBinary`: creates a n-shape binary space多维01空间. Argument n can be a number or a `list` of numbers.

  举例:

  `MultiBinary(5)` 表示5维的0或1的数组。 

  `MultiBinary([3,2])` 表示3x2维的0或1的数组。

- `MultiDiscrete`: consists of a series of `Discrete` action spaces多维离散空间 with a different number of actions in each element.（游戏控制器）

  举例:

  `MultiDiscrete([5,2,2]) `表示三个discrete action space。

实例：

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

可以不用改变源代码，就让我们包装现有环境，把各种新功能包装进现有环境

```python
>>> import gym
>>> from gym.wrappers import RescaleAction

>>> base_env = gym.make("BipedalWalker-v3")						# 1.要包装的基础环境
>>> base_env.action_space										# 显示环境的动作空间
Box([-1. -1. -1. -1.], [1. 1. 1. 1.], (4,), float32)

>>> wrapped_env = RescaleAction(base_env, min_action=0, max_action=1)	# 2.包装一个新的动作空间
>>> wrapped_env.action_space
Box([0. 0. 0. 0.], [1. 1. 1. 1.], (4,), float32)

>>> wrapped_env.unwrapped										# 3.将环境取消包装
```



## 1.4 Playing within environment

```python
from gym.utils.play import play			# 使用play函数
play(gym.make('Pong-v0'))
```



# 2、 Custom environment

[源代码](https://github.com/Farama-Foundation/gym-examples/blob/main/gym_examples/envs/grid_world.py)

[教程](https://www.gymlibrary.dev/content/environment_creation/)

## 2.1 环境信息

The environment consists of a 2-dimensional square grid of fixed size (specified via the `size` parameter during construction). The agent can move vertically or horizontally between grid cells in each timestep. The goal of the agent is to navigate to a target on the grid that has been placed randomly at the beginning of the episode.

- **Observations** provide the location of the target and agent.
- There are 4 **actions** in our environment, corresponding to the movements “right”, “up”, “left”, and “down”.
- A **done** signal is issued as soon as the agent has navigated to the grid cell where the target is located.
- **Rewards** are binary and sparse, meaning that the immediate reward is always zero, unless the agent has reached the target, then it is 1.

### 2.1.1 文件结构：

```shell
(.env) xuy1fe@FE-C-009N1:~/Desktop/Ball_plate/Work_Space/RL-learning/gym-examples$ tree
.
├── gym_examples
│   ├── envs
│   │   ├── grid_world.py
│   │   └── __init__.py
│   ├── __init__.py
│   └── wrappers
│       ├── clip_reward.py
│       ├── discrete_actions.py
│       ├── __init__.py
│       ├── reacher_weighted_reward.py
│       └── relative_position.py
├── gym_examples.egg-info
│   ├── dependency_links.txt
│   ├── PKG-INFO
│   ├── requires.txt
│   ├── SOURCES.txt
│   └── top_level.txt
├── README.md
└── setup.py

```

## 2.2 Declaration and Initialization

```python
import gym
from gym import spaces
import pygame
import numpy as np

# declaration of GridWorldEnv
class GridWorldEnv(gym.Env):
    # 1. *************** metadata attribute用于指定渲染render模式和framrate帧率***************
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
	
    # 每一个环境都应该支持None作为渲染模式
    # 这个环境会接收一个变量size，用于制定方块的尺寸。
    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
		
        # 2. ***************由spaces类定义观察空间***************
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        # 因为观察空间要提供目标和智能体的位置，所以用字典来包含2个实例
        # 每一个实例有2个维度(x,y)，所以shape=(2,) =>元组如果只有1个元素需要加逗号
        # 每一个维度的范围是0 到 size-1.
        self.observation_space = spaces.Dict(
            {	
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )
		
        # 3. *************** 由spaces类定义动作空间***************
        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }
		
        # assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常。
        # assert 触发异常后，后面的代码都不执行。
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
```



## 2.3 Constructing Observations From Environment States

2.2的`class GridWorldEnv(gym.Env)`的成员函数

- _get_obs(): 因为在`reset`和`step`中我们都需要oberservation的信息，所以提供一个私有方法private method来返回observation space的值是很有必要的
- _get_info(): 同样的，因为本环境需要得到agent和target之间的距离，这些需要额外操作的值可以放在这个私有方法里来返回

```python
def _get_obs(self):
    return {"agent": self._agent_location, "target": self._target_location}

def _get_info(self):
    return {
        "distance": np.linalg.norm(
            self._agent_location - self._target_location, ord=1
        )
    }
```

## 2.4 Reset

2.2的`class GridWorldEnv(gym.Env)`的成员函数

- reset(): 被用来重置每一个新的集episode
- reset(): 方法应该返回 初始观察空间 和 其他一些有用的 信息，用2.3的方法得到他们

```python
def reset(self, seed=None, options=None):
    # We need the following line to seed self.np_random
    # 重置随机数种子
    super().reset(seed=seed)
	
    # 本环境中需要重置agent和target的位置
    # Choose the agent's location uniformly at random
    self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
    # We will sample the target's location randomly until it does not coincide with the agent's location
    self._target_location = self._agent_location
    while np.array_equal(self._target_location, self._agent_location):
        self._target_location = self.np_random.integers(
            0, self.size, size=2, dtype=int
        )

    observation = self._get_obs()
    info = self._get_info()

    if self.render_mode == "human":
        self._render_frame()

    return observation, info
```

## 2.5 step

2.2的`class GridWorldEnv(gym.Env)`的成员函数

- step(): 包含环境的大部分逻辑logic。
- step()会接收action,然后计算环境的状态并返回一个4-tuple:`(observation,reward,done,info)`

```python
def step(self, action):
    # Map the action (element of {0,1,2,3}) to the direction we walk in
    # 2.2中定义的方法，将action从0123,映射为上下移动
    direction = self._action_to_direction[action]
    
    # We use `np.clip` to make sure we don't leave the grid
    # numpy.clip(a, a_min, a_max) 将数组a夹在a_min和a_max之间
    self._agent_location = np.clip(
        self._agent_location + direction, 0, self.size - 1
    )
    
    # An episode is done if the agent has reached the target
    terminated = np.array_equal(self._agent_location, self._target_location)
    reward = 1 if terminated else 0  # Binary sparse rewards
    observation = self._get_obs()
    info = self._get_info()

    if self.render_mode == "human":
        self._render_frame()

    return observation, reward, terminated, False, info
```



## 2.6 Rendering

2.2的`class GridWorldEnv(gym.Env)`的成员函数

- 这里我们使用PyGame来渲染。

```python
def render(self):
    if self.render_mode == "rgb_array":
        return self._render_frame()

def _render_frame(self):
    if self.window is None and self.render_mode == "human":
        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode((self.window_size, self.window_size))
    if self.clock is None and self.render_mode == "human":
        self.clock = pygame.time.Clock()

    canvas = pygame.Surface((self.window_size, self.window_size))
    canvas.fill((255, 255, 255))
    pix_square_size = (
        self.window_size / self.size
    )  # The size of a single grid square in pixels

    # First we draw the target
    pygame.draw.rect(
        canvas,
        (255, 0, 0),
        pygame.Rect(
            pix_square_size * self._target_location,
            (pix_square_size, pix_square_size),
        ),
    )
    # Now we draw the agent
    pygame.draw.circle(
        canvas,
        (0, 0, 255),
        (self._agent_location + 0.5) * pix_square_size,
        pix_square_size / 3,
    )

    # Finally, add some gridlines
    for x in range(self.size + 1):
        pygame.draw.line(
            canvas,
            0,
            (0, pix_square_size * x),
            (self.window_size, pix_square_size * x),
            width=3,
        )
        pygame.draw.line(
            canvas,
            0,
            (pix_square_size * x, 0),
            (pix_square_size * x, self.window_size),
            width=3,
        )

    if self.render_mode == "human":
        # The following line copies our drawings from `canvas` to the visible window
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
        self.clock.tick(self.metadata["render_fps"])
    else:  # rgb_array
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )
```

## 2.7 Close

2.2的`class GridWorldEnv(gym.Env)`的成员函数

- 因为`render_mode` may be `"human"` and we might need to close the window that has been opened:

```python
def close(self):
    if self.window is not None:
        pygame.display.quit()
        pygame.quit()
```

## 2.8 Register Env

为了让Gym能识别我们自己设计的环境，我们必须注册他们

参见2.1.1的文件结构，代码在：`gym-examples/gym_examples/__init__.py`.

```python
from gym.envs.registration import register

register(
    id="gym_examples/GridWorld-v0",
    entry_point="gym_examples.envs:GridWorldEnv",
)
```

- 环境id由三个组成：

  - an optional namespace (here: `gym_examples`), 
  - a mandatory name (here: `GridWorld`) 
  - an optional but recommended version (here: v0).

- 除了id和entry_point,还可以传入如下参数到register：

  | Name                | Type    | Default | Description                                                  |
  | ------------------- | ------- | ------- | ------------------------------------------------------------ |
  | `reward_threshold`  | `float` | `None`  | The reward threshold before the task is considered solved    |
  | `nondeterministic`  | `bool`  | `False` | Whether this environment is non-deterministic even after seeding |
  | `max_episode_steps` | `int`   | `None`  | The maximum number of steps that an episode can consist of. If not `None`, a `TimeLimit` wrapper is added |
  | `order_enforce`     | `bool`  | `True`  | Whether to wrap the environment in an `OrderEnforcing` wrapper |
  | `autoreset`         | `bool`  | `False` | Whether to wrap the environment in an `AutoResetWrapper`     |
  | `kwargs`            | `dict`  | `{}`    | The default kwargs to pass to the environment class          |

还需要加入`__init__.py`文件：

- `gym-examples/gym_examples/envs/__init__.py` should have:

  ```python
  from gym_examples.envs.grid_world import GridWorldEnv
  ```

  

## 2.9 Create a package

将我们的代码包装成一个python包

参见2.1.1的文件结构：`gym-examples/setup.py`

```python
from setuptools import setup

setup(
    name="gym_examples",
    version="0.0.1",
    install_requires=["gym==0.26.0", "pygame==2.1.0"],
)
```

- 安装我们的python包

  ```python
  pip install -e gym-examples
  ```

- 然后就可以在python中使用我们的环境了

  ```python
  import gym_examples
  env = gym.make('gym_examples/GridWorld-v0')
  ```

  
