import glob
import imp
from turtle import color
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    for e in tf.train.summary_iterator(file):
    # for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Eval_AverageReturn':
                Y.append(v.simple_value)
    return X, Y

if __name__ == '__main__':
    import glob
    
    # logdir = []
    # logdir.append('data/q1_sb_no_rtg_dsa_CartPole-v0_19-02-2022_16-40-57/events*')
    # logdir.append('data/q1_sb_rtg_dsa_CartPole-v0_19-02-2022_16-47-01/events*')
    # logdir.append('data/q1_sb_rtg_na_CartPole-v0_19-02-2022_16-47-56/events*')
    # logdir.append('data/q1_lb_no_rtg_dsa_CartPole-v0_19-02-2022_16-49-39/events*')
    # logdir.append('data/q1_lb_rtg_dsa_CartPole-v0_19-02-2022_19-26-39/events*')
    # logdir.append('data/q1_lb_rtg_na_CartPole-v0_19-02-2022_19-27-07/events*')
    # colors = ['skyblue','orange','red']
    logdir = 'data/q1_sb_no_rtg_dsa_CartPole-v0_19-02-2022_16-40-57/events*'
    # eventfile = glob.glob(logdir)[0]
    # X = []
    # Y = []
    # iter = np.arange(100)
    # for i,path in enumerate(logdir):
    #     try:
    #         eventfile = glob.glob(path)[0]
    #     except:
    #         print(f"wrong path : {path}")
    #     x, y = get_section_results(eventfile)
    #     X.append(x)
    #     Y.append(y)
    #     if i == 0 or i == 3:
    #         plt.figure()
    #         # plt.title(label = "small batch" if i == 0 else "big batch")
    #     plt.plot(iter,y,color=colors[i%3],label = f"y{i}")
    #     plt.legend()
    #     if i == 2 or i == 5:
    #         plt.show()

    # X, Y = get_section_results(eventfile)
    # for i, (x, y) in enumerate(zip(X, Y)):
    #     print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))