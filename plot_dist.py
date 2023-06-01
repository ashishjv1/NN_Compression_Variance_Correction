

import seaborn as sns
import matplotlib.pyplot as plt

def plot_dist(weight_list, ylim=(0,0.75), xlim=(-0.4, 1.5)):

        linestyle = ['-', '-.', '--', '.']
        color = ['black', 'lightgreen', 'blue', 'red']
        label = ['Original Layer', 'FL1', 'FL2', 'FL3', 'FL4']
        alpha = [0.4, 0.2, 0.2]

        plt.figure(figsize=(12,12), dpi=100)
        for i in range(len(weight_list)):
            ax = sns.kdeplot(weight_list[i][0:1000].flatten().detach().numpy()) 
        plt.ylim(ylim)
        plt.xlim(xlim)
        plt.legend()