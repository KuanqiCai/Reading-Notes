# 一、基本使用

[参考](https://zhuanlan.zhihu.com/p/406517851)

## 1.1 gym环境使用

- 下载环境

  ```shell
  pip install gym
  pip install Box2D
  pip install pyglet==1.5.27 # 如果python版本是3.8以上就可以下最新版
  ```

- 使用环境自带的随机采样

  ```python
  import gym
  
  env_name = "LunarLander-v2"
  env = gym.make(env_name)          # 导入注册器中的环境
  
  episodes = 10
  for episode in range(1, episodes + 1):
      state = env.reset()           # gym风格的env开头都需要reset一下以获取起点的状态
      done = False
      score = 0
  
      while not done:
          env.render()              # 将当前的状态化成一个frame，再将该frame渲染到小窗口上
          action = env.action_space.sample()     # 通过随机采样获取一个随即动作
          n_state, reward, done, info = env.step(action)    # 将动作扔进环境中，从而实现和模拟器的交互
          score += reward
      print("Episode : {}, Score : {}".format(episode, score))
  
  env.close() 
  ```

## 1.2 sb3 使用

- 训练模型

  ```python
  import gym
  from stable_baselines3 import DQN
  from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
  from stable_baselines3.common.evaluation import evaluate_policy
  
  env_name = "LunarLander-v2"
  env = gym.make(env_name)
  
  """
  DummyVecEnv, 该类可以对env进行包装, 从而完成环境的向量化
  向量化的好处:
  如果你有多个env需要使用,那么直接将它们按照上述的格式(无输入到环境类的映射)写成一个列表传入DummyVecEnv中,
  那么在后续训练时，向量化的多个环境便会在同一个线程或者进程中被使用，从而提高采样和训练的效率。
  """ 
  env = DummyVecEnv([lambda : env])
  
  
  """
  "MlpPolicy"定义了DQN的策略网络是一个MLP网络, 也可以填CnnPolicy来定义策略网络为CNN.
  不过此处的输入就是一个8维向量, 没必要做local connection, 所以还是选择MLP就好
  """
  model = DQN(
      "MlpPolicy", 
      env=env,
      verbose=1
  )
  
  model.learn(total_timesteps=1e5)
  ```

- 评估模型

  ```python
  # n_eval_episodes代表评估的次数，该值建议为10-30，越高评估结果越可靠。
  mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
  env.close()
  mean_reward, std_reward
  
  # evaluate_policy返回这若干次测试后模型的得分的均值和方差
  #(-166.24617834256043, 50.530194648006685)
  ```

  



# 二、Tensorboard

使用tensorboard来查看数据并可视化。

[Pytorch中使用tensorboard](https://pytorch.org/docs/stable/tensorboard.html)

[stable_baseline3中使用tensorboard](https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html)

[tensorflow中使用tensorboard](https://www.tensorflow.org/tensorboard/get_started)

## 2.1 记录基本数据

### 2.1.1 基本数据

[基本数据](https://stable-baselines3.readthedocs.io/en/master/common/logger.html#logger)有eval,rollout,time和train

### 2.1.2 记录数据

默认log name是使用的算法名：比如这里的A2C

```python
from stable_baselines3 import A2C

model = A2C("MlpPolicy", "CartPole-v1", verbose=1, tensorboard_log="./a2c_cartpole_tensorboard/")
model.learn(total_timesteps=10_000)
```

也可以自定义算法名：

```python
from stable_baselines3 import A2C

model = A2C("MlpPolicy", "CartPole-v1", verbose=1, tensorboard_log="./a2c_cartpole_tensorboard/")
model.learn(total_timesteps=10_000, tb_log_name="first_run")
# Pass reset_num_timesteps=False to continue the training curve in tensorboard
# By default, it will create a new curve
# Keep tb_log_name constant to have continuous curve (see note below)
model.learn(total_timesteps=10_000, tb_log_name="second_run", reset_num_timesteps=False)
model.learn(total_timesteps=10_000, tb_log_name="third_run", reset_num_timesteps=False)
```

### 2.1.3 使用tensorboard命令查看log

```python
tensorboard --logdir ./a2c_cartpole_tensorboard/
```

也可以添加多个地址：

```python
tensorboard --logdir ./a2c_cartpole_tensorboard/;./ppo2_cartpole_tensorboard/
```

## 2.2 记录其他数据

  继承BaseCallback类来记录更多数据:

```python
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

model = SAC("MlpPolicy", "Pendulum-v1", tensorboard_log="/tmp/sac/", verbose=1)


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value = np.random.random()
        self.logger.record("random_value", value)
        return True


model.learn(50000, callback=TensorboardCallback())
```



## 2.3 记录图像

如果是仿真环境就会有render()函数，用它来获取环境图像

```python
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Image

model = SAC("MlpPolicy", "Pendulum-v1", tensorboard_log="/tmp/sac/", verbose=1)


class ImageRecorderCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        image = self.training_env.render(mode="rgb_array")
        # "HWC" specify the dataformat of the image, here channel last
        # (H for height, W for width, C for channel)
        # See https://pytorch.org/docs/stable/tensorboard.html
        # for supported formats
        self.logger.record("trajectory/image", Image(image, "HWC"), exclude=("stdout", "log", "json", "csv"))
        return True


model.learn(50000, callback=ImageRecorderCallback())
```

## 2.4 记录图表

可以记录matplotlib绘制的图像

```python
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure

model = SAC("MlpPolicy", "Pendulum-v1", tensorboard_log="/tmp/sac/", verbose=1)


class FigureRecorderCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        # Plot values (here a random variable)
        figure = plt.figure()
        figure.add_subplot().plot(np.random.random(3))
        # Close the figure after logging it
        self.logger.record("trajectory/figure", Figure(figure, close=True), exclude=("stdout", "log", "json", "csv"))
        plt.close()
        return True


model.learn(50000, callback=FigureRecorderCallback())
```

## 2.5 记录视频

要下载[moviepy](https://zulko.github.io/moviepy/) 

```python
from typing import Any, Dict

import gym
import torch as th

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video


class VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env: gym.Env, render_freq: int, n_eval_episodes: int = 1, deterministic: bool = True):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            screens = []

            def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                """
                Renders the environment in its current state, recording the screen in the captured `screens` list

                :param _locals: A dictionary containing all local variables of the callback's scope
                :param _globals: A dictionary containing all global variables of the callback's scope
                """
                screen = self._eval_env.render(mode="rgb_array")
                # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                screens.append(screen.transpose(2, 0, 1))

            evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )
            self.logger.record(
                "trajectory/video",
                Video(th.ByteTensor([screens]), fps=40),
                exclude=("stdout", "log", "json", "csv"),
            )
        return True


model = A2C("MlpPolicy", "CartPole-v1", tensorboard_log="runs/", verbose=1)
video_recorder = VideoRecorderCallback(gym.make("CartPole-v1"), render_freq=5000)
model.learn(total_timesteps=int(5e4), callback=video_recorder)
```

## 2.6 记录超参

```python
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam


class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0.0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True


model = A2C("MlpPolicy", "CartPole-v1", tensorboard_log="runs/", verbose=1)
model.learn(total_timesteps=int(5e4), callback=HParamCallback())
```



## 2.7 直接使用Pytorch来记录

直接使用pytorch的SummaryWritter来记录数据，SB3官方不推荐此方法。

```python
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat



model = SAC("MlpPolicy", "Pendulum-v1", tensorboard_log="/tmp/sac/", verbose=1)


class SummaryWriterCallback(BaseCallback):

    def _on_training_start(self):
        self._log_freq = 1000  # log every 1000 calls

        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self) -> bool:
        if self.n_calls % self._log_freq == 0:
            # You can have access to info from the env using self.locals.
            # for instance, when using one env (index 0 of locals["infos"]):
            # lap_count = self.locals["infos"][0]["lap_count"]
            # self.tb_formatter.writer.add_scalar("train/lap_count", lap_count, self.num_timesteps)

            self.tb_formatter.writer.add_text("direct_access", "this is a value", self.num_timesteps)
            self.tb_formatter.writer.flush()


model.learn(50000, callback=SummaryWriterCallback())
```

