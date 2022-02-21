#### 细节处理

细节处理主要在于如何完整实现policy gradient。在hw1中，我们每次抽取的是batch_size，但这样并不太妥当。因为实际抽取的量往往比batch_size要大一些（当batch_size不是ep_len的整数倍，或者即便是整数倍而有rollout提前终止时）；那么这会导致抽取的batch_size里面有不是完整的rollout，这对估算来说是有误差的。

实际上hw2中给出的处理是，每次训练时抽取总step大于batch_size的完整rollouts——那么这和collect data时候的策略就是一样的了。

此外，注意到我们对reward要做discount，所以必然是要知道哪些reward是在一个rollout里面的。所以这里采取的obs和rews的存储方式是有差异的，obs是一个list，里面是一个个的ob，rews也是一个list，里面是一条条的rollout，而rollout里面才是一个个的reward。然后每个rollout的reward处理完后再`np.concatenate`，达到和obs一致的维度。

对于baseline方法，这时候实际上还没有讲到q function和value function那边，所以这里是用了一个朴素的模拟方法。用一个baseline网络来做，这个网络给的结果是$V_{\phi}^\pi(S_t)$，但虽然这里下标写的是$S_t$，但实际上输入就是obs，obs本身是不带t的标志的。那么自然地就会遇到一个时间上的问题，如果这个路径不是很长，那么S在前面出现和相同的S在后面出现，值很可能是不一样的。那么这个问题在连续空间状态的情况下应该问题不大，是不会前后重复的。另一个我可以想到的解释是，由于比如动作空间是离散的（或者即使动作是连续的），刚开始的几步所能够覆盖的State的范围本身就是有限的，也就是说，一个状态，是有其最早能到到的时间的。每次策略也只需要在同时能够达到的几个状态中选择即可。——总而言之，从实践的角度，这么做效果没问题。

#### correction

之前参考的https://github.com/vincentkslim/cs285_homework_fall2020里面，我发现一个错误：`targets = normalize(q_values, np.mean(q_values), np.mean(q_values))`。对baseline做更新的时候，我们是有做normalize的（baseline netword的输出是被训练为normalize之后的），而这里直接不小心以mean为std，对效果有所影响。

#### 耗时

如作业描述中所说，前两个实验都很快，第三个（用cpu，8259u）要1~2h。第四个没跑。
