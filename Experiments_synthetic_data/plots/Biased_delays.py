import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rc('pdf', fonttype=42)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
from matplotlib.ticker import FuncFormatter


Fontsize=28
Labelsize=23
Legendsize=18
notationsize = 18



def Simulation_Biased_delays():
    data1 = np.loadtxt('./data/Cum_RR_Biased_delays.txt')
    data2 = np.loadtxt('./data/Cum_FR_Biased_delays.txt')
    round = np.array(list(range(len(data1[0]))))
    fig = plt.figure(figsize=(8,12))
    plt.subplot(2,1,1)
    ax=plt.gca()
    def formatnum1(x, pos):
        return '$%d$' % (x/10000)
    def formatnum2(x, pos):
        return '$%d$' % (x/1000)
    plt.annotate(r'$\times 10^{4}$',(40000,0),(37800,6800*(150/5900)),fontsize = notationsize)
    plt.annotate(r'$\times 10^{3}$',(0,5800),(-5,6800*(5400/5900)),fontsize = notationsize)
    formatter1 = FuncFormatter(formatnum1)
    formatter2 = FuncFormatter(formatnum2)
    ax.xaxis.set_major_formatter(formatter1)
    ax.yaxis.set_major_formatter(formatter2)

    plt.plot(round, data1[0], label = 'OP-FCUCB-D',color=(0.5490196078431373, 0.33725490196078434, 0.29411764705882354),linewidth=2,marker='^', markersize=9, markevery=3200)
    plt.plot(round, data1[1], label = 'OP-FCTS-D',color=(0.17254901960784313, 0.6274509803921569, 0.17254901960784313),linewidth=2,marker='*',markersize=11, markevery=3200)
    plt.plot(round, data1[2], label = 'FCUCB-D',color=(0.09019607843137255, 0.7450980392156863, 0.8117647058823529),linewidth=2,marker='v', markersize=9, markevery=3200)
    plt.plot(round, data1[3], label = 'FCTS-D',color=(1.0, 0.4980392156862745, 0.054901960784313725),linewidth=2,marker='s',markersize=9, markevery=3200)
    plt.plot(round, data1[4], label = 'CUCB-D',color=(0.8392156862745098, 0.15294117647058825, 0.1568627450980392),linewidth=2,marker='p', markersize=9, markevery=3200)
    plt.plot(round, data1[5], label = 'MP-TS-D',color=(0.5803921568627451, 0.403921568627451, 0.7411764705882353),linewidth=2,marker='h',markersize=9, markevery=3200)
    plt.plot(round, data1[6], label = 'FGreedy-D',color='goldenrod',linewidth=2,marker='.',markersize=15, markevery=3200)
    plt.tick_params(labelsize=Labelsize)
    plt.grid(alpha=0.2)
    plt.legend(fontsize=Legendsize,loc = 'lower center',bbox_to_anchor=(0.58, 0),framealpha=0.3,ncol=2,columnspacing=0.5)
    plt.ylabel("Reward regret",fontsize=Fontsize)
    plt.xlim(0,None)
    plt.ylim(0,6800)
    plt.xticks(np.arange(0, 40001, 10000))



    plt.subplot(2,1,2)
    ax=plt.gca()
    def formatnum(x, pos):
        return '$%d$' % (x/10000)
    plt.annotate(r'$\times 10^{4}$',(0,32000),(-5,36000*(5400/5900)),fontsize = notationsize)
    plt.annotate(r'$\times 10^{4}$',(40000,0),(37800,36000*(150/5900)),fontsize = notationsize)
    formatter = FuncFormatter(formatnum)
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_formatter(formatter)
    plt.plot(round, data2[0], label = 'OP-FCUCB-D',color=(0.5490196078431373, 0.33725490196078434, 0.29411764705882354),linewidth=2,marker='^', markersize=9, markevery=3200)
    plt.plot(round, data2[1], label = 'OP-FCTS-D',color=(0.17254901960784313, 0.6274509803921569, 0.17254901960784313),linewidth=2,marker='*',markersize=11, markevery=3200)
    plt.plot(round, data2[2], label = 'FCUCB-D',color=(0.09019607843137255, 0.7450980392156863, 0.8117647058823529),linewidth=2,marker='v', markersize=9, markevery=3200)
    plt.plot(round, data2[3], label = 'FCTS-D',color=(1.0, 0.4980392156862745, 0.054901960784313725),linewidth=2,marker='s',markersize=9, markevery=3200)
    plt.plot(round, data2[4], label = 'CUCB-D',color=(0.8392156862745098, 0.15294117647058825, 0.1568627450980392),linewidth=2,marker='p', markersize=9, markevery=3200)
    plt.plot(round, data2[5], label = 'MP-TS-D',color=(0.5803921568627451, 0.403921568627451, 0.7411764705882353),linewidth=2,marker='h',markersize=9, markevery=3200)
    plt.plot(round, data2[6], label = 'FGreedy-D',color='goldenrod',linewidth=2,marker='.',markersize=15, markevery=3200)
    plt.tick_params(labelsize=Labelsize)
    plt.grid(alpha=0.2)
    plt.legend(fontsize=Legendsize,loc = 'lower center',bbox_to_anchor=(0.6, 0),framealpha=0.3,ncol=2,columnspacing=0.5)
    plt.xlabel("Round",fontsize=Fontsize)
    plt.ylabel("Fairness regret",fontsize=Fontsize)
    plt.xlim(0,None)
    plt.ylim(0,36000)
    plt.xticks(np.arange(0, 40001, 10000))
    plt.tight_layout()
    fig.align_ylabels()
    plt.show()