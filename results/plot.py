
import os
import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns; sns.set()
import glob2
import glob
import argparse
import seaborn as sns; sns.set()
import pandas as pd

sns.set_style('darkgrid')
import matplotlib
# matplotlib.use('TkAgg') # Can change to 'Agg' for non-interactive mode
matplotlib.rcParams.update({'font.size': 15})

def smooth_curve(x, y, window):
    halfwidth = int(np.ceil(len(x) / window))  # Halfwidth of our smoothing convolution

    #print(halfwidth)
    k = halfwidth
    xsmoo = x
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='same') / np.convolve(np.ones_like(y), np.ones(2 * k + 1),
        mode='same')
    # print(xsmoo, ysmoo)
    return xsmoo, ysmoo


def pad(xs, value=np.nan):
    maxlen = np.max([len(x) for x in xs])
    padded_xs = []
    for x in xs:
        if x.shape[0] >= maxlen:
            padded_xs.append(x)

        padding = np.ones((maxlen - x.shape[0],) + x.shape[1:]) * value
        x_padded = np.concatenate([x, padding], axis=0)
        assert x_padded.shape[1:] == x.shape[1:]
        assert x_padded.shape[0] == maxlen
        padded_xs.append(x_padded)
    return np.array(padded_xs)


def plot_baselines_return(env_name):
    result_path = './{}-test_return.csv'.format(env_name)
    algs = ['wtd','iql', 'awac','td3bc','wgcsl','gcsl']
    keys = ['method: {} - Test/undiscounted_return'.format(alg) for alg in algs ]
    min = ['method: {} - Test/undiscounted_return__MIN'.format(alg) for alg in algs]
    max = ['method: {} - Test/undiscounted_return__MAX'.format(alg) for alg in algs]
    # algs = ['HER_(0-10cm)','HER_(10-20cm)','HER_(20-40cm)']
    #colors =['r', 'b', 'g', '#9467bd']
    colors = [
        '#d62728',  # brick red
        '#1f77b4',  # muted blue
        '#2ca02c',  # cooked asparagus green
        '#ff7f0e',  # safety orange
        '#e377c2',  # raspberry yogurt pink
        '#9370DB',  # light green
        # '#d62728',  # brick red
        'slategray',
        # '#e377c2',  # raspberry yogurt pink
        '#7f7f7f',  # middle gray
        '#bcbd22',  # curry yellow-green
        '#17becf'  # blue-teal
    ]
    linestyles = ['-', '-.', ':', '--', '--', '-', '--']
    linestyles = ['-', '-', '-', '-','-', '-','-', '-']

    #data = load_results('../results/FetchPnPObstacle-v1/OMEGA/progress_444_explor_ratio_0.5_03_000.csv')
    plt.figure(figsize=(8, 6))
    sns.despine(left=True, bottom=True)
    data = pd.read_csv(result_path, delimiter=',')
    for i in range(len(algs)):
        xs = np.array(data['Step']).flatten()
        y_mean = np.array(data[keys[i]]).flatten()
        xs, y_mean = smooth_curve(xs, y_mean, window=35)
        # smooth_value.append(y)
        y_lower = np.array(data[min[i]]).flatten()
        xs, y_lower = smooth_curve(xs, y_lower,window=20)
        y_upper = np.array(data[max[i]]).flatten()
        xs, y_upper = smooth_curve(xs, y_upper,window=20)
        assert xs.shape == y_lower.shape == y_upper.shape
        if algs[i] == 'wtd':
            plt.plot(xs, y_mean, label='Ours', color=colors[i], linewidth=2.5, linestyle=linestyles[i])
        else:
            plt.plot(xs, y_mean, label=algs[i].upper(), color=colors[i], linewidth=2.5, linestyle=linestyles[i])
        plt.fill_between(xs, y_lower, y_upper, color=colors[i], alpha=0.15)
    plt.xlabel('Epoch',fontdict=dict(fontsize=14))
    plt.ylabel('Average Discount Returns',fontdict=dict(fontsize=15))
    plt.xticks(np.arange(0, 101, 20), fontproperties='Times New Roman', size=14)
    if env_name =='fetchslide':
        plt.yticks(np.arange(0, 21, 5),fontproperties='Times New Roman', size=12)
    elif env_name=='fetchpush' or 'fetchpick':
        plt.yticks(np.arange(0, 41, 10),fontproperties='Times New Roman', size=12)
    # else:
    #     plt.yticks(np.arange(0, 30, 5),fontproperties='Times New Roman', size=12)
    plt.legend(fancybox=True)
    # if env_name == 'fetchpick':
    #     plt.title('FetchPickAndPlace',fontdict=dict(fontsize=15))
    # elif env_name == 'fetchpush':
    #     plt.title('FetchPush',fontdict=dict(fontsize=15))
    # elif env_name == 'fetchslide':
    #     plt.title('FetchSlide',fontdict=dict(fontsize=15))
    # elif env_name =='handreach':
    #     plt.title('HandReach',fontdict=dict(fontsize=15))

    plt.xlim(0, 100)
  
    save_dir = './figures'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'fig_{}_return.png'.format(env_name)), dpi=400, bbox_inches='tight')




def plot_baselines_success_rate(env_name):
    result_path = './{}-test_success_rate.csv'.format(env_name)
    algs = ['wtd','iql', 'awac','td3bc','wgcsl','gcsl']
    keys = ['method: {} - Test/success_rate'.format(alg) for alg in algs ]
    min = ['method: {} - Test/success_rate__MIN'.format(alg) for alg in algs]
    max = ['method: {} - Test/success_rate__MAX'.format(alg) for alg in algs]
    # algs = ['HER_(0-10cm)','HER_(10-20cm)','HER_(20-40cm)']
    #colors =['r', 'b', 'g', '#9467bd']
    colors = [
        '#d62728',  # brick red
        '#1f77b4',  # muted blue
        '#2ca02c',  # cooked asparagus green
        '#ff7f0e',  # safety orange
        '#e377c2',  # raspberry yogurt pink
        '#9370DB',  # light green
        # '#d62728',  # brick red
        'slategray',
        # '#e377c2',  # raspberry yogurt pink
        '#7f7f7f',  # middle gray
        '#bcbd22',  # curry yellow-green
        '#17becf'  # blue-teal
    ]
    linestyles = ['-', '-.', ':', '--', '--', '-', '--']
    linestyles = ['-', '-', '-', '-','-', '-','-', '-']

    #data = load_results('../results/FetchPnPObstacle-v1/OMEGA/progress_444_explor_ratio_0.5_03_000.csv')
    plt.figure(figsize=(8, 6))
    sns.despine(left=True, bottom=True)
    data = pd.read_csv(result_path, delimiter=',')
    for i in range(len(algs)):
        xs = np.array(data['Step']).flatten()
        y_mean = np.array(data[keys[i]]).flatten()
        xs, y_mean = smooth_curve(xs, y_mean, window=35)
        # smooth_value.append(y)
        y_lower = np.array(data[min[i]]).flatten()
        xs, y_lower = smooth_curve(xs, y_lower,window=20)
        y_upper = np.array(data[max[i]]).flatten()
        xs, y_upper = smooth_curve(xs, y_upper,window=20)
        assert xs.shape == y_lower.shape == y_upper.shape
        if algs[i] == 'wtd':
            plt.plot(xs, y_mean, label='Ours', color=colors[i], linewidth=2.5, linestyle=linestyles[i])
        else:
            plt.plot(xs, y_mean, label=algs[i].upper(), color=colors[i], linewidth=2.5, linestyle=linestyles[i])
        plt.fill_between(xs, y_lower, y_upper, color=colors[i], alpha=0.15)
    plt.xlabel('Epoch',fontdict=dict(fontsize=14))
    plt.ylabel('Average Test Success Rate',fontdict=dict(fontsize=15))
    plt.xticks(np.arange(0, 101, 20), fontproperties='Times New Roman', size=12)
    if env_name =='fetchslide':
        plt.yticks(np.arange(0, 0.5, 0.1),fontproperties='Times New Roman', size=12)
    elif env_name =='handreach':
        plt.yticks(np.arange(0, 0.6, 0.1),fontproperties='Times New Roman', size=12)
    else:
        plt.yticks(np.arange(0, 1, 0.2),fontproperties='Times New Roman', size=12)
    plt.legend(fancybox=True)
    # if env_name == 'fetchpick':
    #     plt.title('FetchPickAndPlace',fontdict=dict(fontsize=15))
    # elif env_name == 'fetchpush':
    #     plt.title('FetchPush',fontdict=dict(fontsize=15))
    # elif env_name == 'fetchslide':
    #     plt.title('FetchSlide',fontdict=dict(fontsize=15))
    # elif env_name =='handreach':
    #     plt.title('HandReach',fontdict=dict(fontsize=15))

    plt.xlim(0, 100)
  
    save_dir = './figures'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'fig_{}_test_success_rate.png'.format(env_name)), dpi=400, bbox_inches='tight')


def plot_baselines_final_distance(env_name):
    result_path = './{}-test_finaldistance.csv'.format(env_name)
    algs = ['wtd','iql', 'awac','td3bc','wgcsl','gcsl']
    keys = ['method: {} - Test/final_distance'.format(alg) for alg in algs ]
    min = ['method: {} - Test/final_distance__MIN'.format(alg) for alg in algs]
    max = ['method: {} - Test/final_distance__MAX'.format(alg) for alg in algs]
    # algs = ['HER_(0-10cm)','HER_(10-20cm)','HER_(20-40cm)']
    #colors =['r', 'b', 'g', '#9467bd']
    colors = [
        '#d62728',  # brick red
        '#1f77b4',  # muted blue
        '#2ca02c',  # cooked asparagus green
        '#ff7f0e',  # safety orange
        '#e377c2',  # raspberry yogurt pink
        '#9370DB',  # light green
        # '#d62728',  # brick red
        'slategray',
        # '#e377c2',  # raspberry yogurt pink
        '#7f7f7f',  # middle gray
        '#bcbd22',  # curry yellow-green
        '#17becf'  # blue-teal
    ]
    linestyles = ['-', '-.', ':', '--', '--', '-', '--']
    linestyles = ['-', '-', '-', '-','-', '-','-', '-']

    #data = load_results('../results/FetchPnPObstacle-v1/OMEGA/progress_444_explor_ratio_0.5_03_000.csv')
    plt.figure(figsize=(8, 6))
    sns.despine(left=True, bottom=True)
    data = pd.read_csv(result_path, delimiter=',')
    for i in range(len(algs)):
        xs = np.array(data['Step']).flatten()
        y_mean = np.array(data[keys[i]]).flatten()
        xs, y_mean = smooth_curve(xs, y_mean, window=35)
        # smooth_value.append(y)
        y_lower = np.array(data[min[i]]).flatten()
        xs, y_lower = smooth_curve(xs, y_lower,window=20)
        y_upper = np.array(data[max[i]]).flatten()
        xs, y_upper = smooth_curve(xs, y_upper,window=20)
        assert xs.shape == y_lower.shape == y_upper.shape
        if algs[i] == 'wtd':
            plt.plot(xs, y_mean, label='Ours', color=colors[i], linewidth=2.5, linestyle=linestyles[i])
        else:
            plt.plot(xs, y_mean, label=algs[i].upper(), color=colors[i], linewidth=2.5, linestyle=linestyles[i])
        plt.fill_between(xs, y_lower, y_upper, color=colors[i], alpha=0.15)
    plt.xlabel('Epoch',fontdict=dict(fontsize=14))
    plt.ylabel('Average Test Final Distance',fontdict=dict(fontsize=15))
    plt.xticks(np.arange(0, 101, 20), fontproperties='Times New Roman', size=12)
    plt.legend(fancybox=True)
    # if env_name == 'fetchpick':
    #     plt.title('FetchPickAndPlace',fontdict=dict(fontsize=15))
    # elif env_name == 'fetchpush':
    #     plt.title('FetchPush',fontdict=dict(fontsize=15))
    # elif env_name == 'fetchslide':
    #     plt.title('FetchSlide',fontdict=dict(fontsize=15))
    # elif env_name =='handreach':
    #     plt.title('HandReach',fontdict=dict(fontsize=15))

    plt.xlim(0, 100)
  
    save_dir = './figures'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'fig_{}_test_final_distance.png'.format(env_name)), dpi=400, bbox_inches='tight')

path = '/data1/ydy/RL/GoFAR/results/fetchpick-test_discount_return.csv'
results = pd.read_csv(path, delimiter=',')
print(results['method: wtd - Test/discounted_return'])
algs = ['wtd','iql', 'awac','td3bc','wgcsl','gcsl']
return_keys = ['method: {} - Test/discounted_return'.format(alg) for alg in algs ]
print(return_keys)
# plot_baselines_return(env_name='handreach')
# plot_baselines_return(env_name='fetchpush')
plot_baselines_return(env_name='fetchpick')
plot_baselines_return(env_name='fetchpush')
# plot_baselines_success_rate(env_name='handreach')
# plot_baselines_final_distance(env_name='handreach')