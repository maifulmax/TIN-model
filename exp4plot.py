# encoding=utf-8
import matplotlib.pyplot as plt
import pandas as pd
from pylab import *   #支持中文
from matplotlib import rcParams
rcParams['font.family']='sans-serif'
rcParams['font.sans-serif']=['Calibri']


if __name__ == '__main__':
    datafile = './datasets/TIN/result_with_task_range.csv'
    image_file = './datasets/TIN/result_with_task_range.pdf'
    x = list(pd.read_csv(datafile)['num_task'])

    mean = list(pd.read_csv(datafile)['simple'])
    var = list(pd.read_csv(datafile)['simple_var'])
    plt.errorbar(x, mean, var, capsize=4, capthick=1, label=u'TIN-simple')

    mean = list(pd.read_csv(datafile)['task_community'])
    var = list(pd.read_csv(datafile)['task_community_var'])
    plt.errorbar(x, mean, var, capsize=4, capthick=1, label=u'TIN-t')

    mean = list(pd.read_csv(datafile)['worker_community'])
    var = list(pd.read_csv(datafile)['worker_community_var'])
    plt.errorbar(x, mean, var, capsize=4, capthick=1, label=u'TIN-w')

    mean = list(pd.read_csv(datafile)['full'])
    var = list(pd.read_csv(datafile)['full_var'])
    plt.errorbar(x, mean, var, capsize=4, capthick=1, label=u'TIN')

    #plt.xticks(x, fontsize=12)
    plt.xticks(x)
    #plt.yticks(fontsize=12)

    plt.legend()
    plt.tick_params(top=True, bottom=True, left=True, right=True)

    #plt.tick_params(direction='in')

    plt.grid(axis='y', alpha=0.3)
    plt.grid(axis='x', alpha=0.3)
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel('number of task', size=14)  # X轴标签
    plt.ylabel('Accuracy', size=14)  # Y轴标签
    plt.gcf().savefig(image_file, format='pdf')
    plt.show()

    datafile = './datasets/TIN/result_with_worker_range.csv'
    image_file = './datasets/TIN/result_with_worker_range.pdf'
    x = list(pd.read_csv(datafile)['num_worker'])

    mean = list(pd.read_csv(datafile)['simple'])
    var = list(pd.read_csv(datafile)['simple_var'])
    plt.errorbar(x, mean, var, capsize=4, capthick=1, label=u'TIN-simple')

    mean = list(pd.read_csv(datafile)['task_community'])
    var = list(pd.read_csv(datafile)['task_community_var'])
    plt.errorbar(x, mean, var, capsize=4, capthick=1, label=u'TIN-t')

    mean = list(pd.read_csv(datafile)['worker_community'])
    var = list(pd.read_csv(datafile)['worker_community_var'])
    plt.errorbar(x, mean, var, capsize=4, capthick=1, label=u'TIN-w')

    mean = list(pd.read_csv(datafile)['full'])
    var = list(pd.read_csv(datafile)['full_var'])
    plt.errorbar(x, mean, var, capsize=4, capthick=1, label=u'TIN')

    # plt.xticks(x, fontsize=12)
    plt.xticks(x)
    # plt.yticks(fontsize=12)

    plt.legend()
    plt.tick_params(top=True, bottom=True, left=True, right=True)

    # plt.tick_params(direction='in')

    plt.grid(axis='y', alpha=0.3)
    plt.grid(axis='x', alpha=0.3)
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel('number of worker', size=14)  # X轴标签
    plt.ylabel('Accuracy', size=14)  # Y轴标签
    plt.gcf().savefig(image_file, format='pdf')
    plt.show()

    datafile = './datasets/TIN/result_with_domain_range.csv'
    image_file = './datasets/TIN/result_with_domain_range.pdf'
    x = list(pd.read_csv(datafile)['num_domain'])

    mean = list(pd.read_csv(datafile)['simple'])
    var = list(pd.read_csv(datafile)['simple_var'])
    plt.errorbar(x, mean, var, capsize=4, capthick=1, label=u'TIN-simple')

    mean = list(pd.read_csv(datafile)['task_community'])
    var = list(pd.read_csv(datafile)['task_community_var'])
    plt.errorbar(x, mean, var, capsize=4, capthick=1, label=u'TIN-t')

    mean = list(pd.read_csv(datafile)['worker_community'])
    var = list(pd.read_csv(datafile)['worker_community_var'])
    plt.errorbar(x, mean, var, capsize=4, capthick=1, label=u'TIN-w')

    mean = list(pd.read_csv(datafile)['full'])
    var = list(pd.read_csv(datafile)['full_var'])
    plt.errorbar(x, mean, var, capsize=4, capthick=1, label=u'TIN')

    # plt.xticks(x, fontsize=12)
    plt.xticks(x)
    # plt.yticks(fontsize=12)

    plt.legend()
    plt.tick_params(top=True, bottom=True, left=True, right=True)

    # plt.tick_params(direction='in')

    plt.grid(axis='y', alpha=0.3)
    plt.grid(axis='x', alpha=0.3)
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel('number of domain', size=14)  # X轴标签
    plt.ylabel('Accuracy', size=14)  # Y轴标签
    plt.gcf().savefig(image_file, format='pdf')
    plt.show()

    datafile = './datasets/TIN/result_with_redundancy_range.csv'
    image_file = './datasets/TIN/result_with_redundancy_range.pdf'
    x = list(pd.read_csv(datafile)['num_redundancy'])

    mean = list(pd.read_csv(datafile)['simple'])
    var = list(pd.read_csv(datafile)['simple_var'])
    plt.errorbar(x, mean, var, capsize=4, capthick=1, label=u'TIN-simple')

    mean = list(pd.read_csv(datafile)['task_community'])
    var = list(pd.read_csv(datafile)['task_community_var'])
    plt.errorbar(x, mean, var, capsize=4, capthick=1, label=u'TIN-t')

    mean = list(pd.read_csv(datafile)['worker_community'])
    var = list(pd.read_csv(datafile)['worker_community_var'])
    plt.errorbar(x, mean, var, capsize=4, capthick=1, label=u'TIN-w')

    mean = list(pd.read_csv(datafile)['full'])
    var = list(pd.read_csv(datafile)['full_var'])
    plt.errorbar(x, mean, var, capsize=4, capthick=1, label=u'TIN')

    # plt.xticks(x, fontsize=12)
    plt.xticks(x)
    # plt.yticks(fontsize=12)

    plt.legend()
    plt.tick_params(top=True, bottom=True, left=True, right=True)

    # plt.tick_params(direction='in')

    plt.grid(axis='y', alpha=0.3)
    plt.grid(axis='x', alpha=0.3)
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel('number of redundancy', size=14)  # X轴标签
    plt.ylabel('Accuracy', size=14)  # Y轴标签
    plt.gcf().savefig(image_file, format='pdf')
    plt.show()


