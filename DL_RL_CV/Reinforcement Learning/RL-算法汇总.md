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

## 2. Policy Optimization



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