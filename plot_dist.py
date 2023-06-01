

import seaborn as sns
import matplotlib.pyplot as plt

def plot_dist(weight_list, ylim=(0,0.75), xlim=(-0.4, 1.5), fill=False, alpha=0.6):

        linestyle = ['-', '-.', '--', '.']
        color = ['lightblue', 'lightgreen', 'blue', 'red']
        label = ['Original Layer', 'FL1', 'FL2', 'FL3', 'FL4']

        plt.figure(figsize=(8,8), dpi=100)
        for i in range(len(weight_list)):
            ax = sns.kdeplot(weight_list[i][0:1000].flatten().detach().numpy(), fill=fill, alpha=alpha,
                              linewidth=1.1, linestyle=linestyle[i], edgecolor="black", color=color[i], label=label[i]) 
        plt.ylim(ylim)
        plt.xlim(xlim)
        plt.legend()