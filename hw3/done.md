#### 技巧

- torch.gather用来搜寻Tensor中对应位置的值。

- torch.tensor.detach()用来做不要求gradient（因为DDQN做的时候不会求导到用当前网络找到最佳的action的过程）

- replay_buffer：obs和next_obs用同一个序列。注意到有done序列的存在，是可以知道path ends在哪里的。
  - 此外，obs可能包括多个frame，但是在replay buffer中每个frame只存储一次，只是在取出obs的时候找的是一串frame。replay buffer的这些操作都有效地在相同空间增加了能记录的transition的数量。
- 

#### 任务列

1. 跑Pacman验证代码正确

2. 跑Lunar说明DDQN比一般的DQN好
3. 参数寻优
4. （前面是DQN，下面是AC）

