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


def Simulation_Alpha_Pareto_delays_Different_c():
    data1 = np.loadtxt('./data/Cum_RR_Pareto_delays_Different_c.txt')
    data2 = np.loadtxt('./data/Cum_FR_Pareto_delays_Different_c.txt')
    round = np.array(list(range(len(data1[0]))))
    
    fig = plt.figure(figsize=(8,12))
    plt.subplot(2,1,1)
    ax=plt.gca()
    def formatnum1(x, pos):
        return"{:g}".format(x/10000)

    def formatnum2(x, pos):
        return "{:g}".format(x/100)
    plt.annotate(r'$\times 10^{4}$',(20000,0),(18800,170*(150/5900)),fontsize = notationsize)
    plt.annotate(r'$\times 10^{2}$',(0,110),(-5,170*(5400/5900)),fontsize = notationsize)
    formatter1 = FuncFormatter(formatnum1)
    formatter2 = FuncFormatter(formatnum2)
    ax.xaxis.set_major_formatter(formatter1)
    ax.yaxis.set_major_formatter(formatter2)

    plt.plot(round, data1[0], label = 'FCUCB-D',color=(0.09019607843137255, 0.7450980392156863, 0.8117647058823529),linewidth=2,marker='v',markersize=9, markevery=1600)
    plt.plot(round, data1[1], label = 'FCTS-D',color=(1.0, 0.4980392156862745, 0.054901960784313725),linewidth=2,marker='s', markersize=9, markevery=1600)
    plt.plot(round, data1[2], label = 'CUCB-D',color=(0.8392156862745098, 0.15294117647058825, 0.1568627450980392),linewidth=2,marker='p', markersize=9, markevery=1600)
    plt.plot(round, data1[3], label = 'MP-TS-D',color=(0.5803921568627451, 0.403921568627451, 0.7411764705882353),linewidth=2,marker='h',markersize=9, markevery=1600)
    plt.plot(round, data1[4], label = 'FGreedy-D',color='goldenrod',linewidth=2,marker='.', markersize=15, markevery=1600)
    plt.tick_params(labelsize=Labelsize)
    plt.grid(alpha=0.2)
    plt.legend(fontsize=Legendsize,loc = 'upper left',framealpha=0.4,bbox_to_anchor=(0, 0.9))
    plt.ylabel("Reward regret",fontsize=Fontsize)
    plt.xlim(0,None)
    plt.ylim(0,170)
    plt.xticks(np.arange(0, 20001, 5000))
    
    plt.subplot(2,1,2)
    ax=plt.gca()
    def formatnum1(x, pos):
        return"{:g}".format(x/10000)

    def formatnum2(x, pos):
        return "{:g}".format(x/1000)
    plt.annotate(r'$\times 10^{4}$',(20000,0),(18800,2800*(150/5900)),fontsize = notationsize)
    plt.annotate(r'$\times 10^{3}$',(0,110),(-5,2800*(5400/5900)),fontsize = notationsize)
    formatter1 = FuncFormatter(formatnum1)
    formatter2 = FuncFormatter(formatnum2)
    ax.xaxis.set_major_formatter(formatter1)
    ax.yaxis.set_major_formatter(formatter2)
    plt.plot(round, data2[0], label = 'FCUCB-D',color=(0.09019607843137255, 0.7450980392156863, 0.8117647058823529),linewidth=2,marker='v',markersize=9, markevery=1600)
    plt.plot(round, data2[1], label = 'FCTS-D',color=(1.0, 0.4980392156862745, 0.054901960784313725),linewidth=2,marker='s', markersize=9, markevery=1600)
    plt.plot(round, data2[2], label = 'CUCB-D',color=(0.8392156862745098, 0.15294117647058825, 0.1568627450980392),linewidth=2,marker='p', markersize=9, markevery=750)
    plt.plot(round, data2[3], label = 'MP-TS-D',color=(0.5803921568627451, 0.403921568627451, 0.7411764705882353),linewidth=2,marker='h',markersize=9, markevery=750)
    plt.plot(round, data2[4], label = 'FGreedy-D',color='goldenrod',linewidth=2,marker='.',markersize=15, markevery=1600)
    plt.tick_params(labelsize=Labelsize)
    plt.grid(alpha=0.2)
    plt.legend(fontsize=Legendsize,loc = 'upper left',framealpha=0.4,bbox_to_anchor=(0, 0.9))
    plt.xlabel("Round",fontsize=Fontsize)
    plt.ylabel("Fairness regret",fontsize=Fontsize)
    plt.xlim(0,None)
    plt.ylim(0,2800)
    plt.xticks(np.arange(0, 20001, 5000))
    plt.tight_layout()
    fig.align_ylabels() 
    plt.show()


    fig = plt.figure(figsize=(8,12))
    plt.subplot(2,1,1)
    ax=plt.gca()
    def formatnum1(x, pos):
        return"{:g}".format(x/10000)

    def formatnum2(x, pos):
        return "{:g}".format(x/10)
    plt.annotate(r'$\times 10^{4}$',(20000,0),(18800,50*(150/5900)),fontsize = notationsize)
    plt.annotate(r'$\times 10^{1}$',(0,40),(-5,50*(5400/5900)),fontsize = notationsize)
    formatter1 = FuncFormatter(formatnum1)
    formatter2 = FuncFormatter(formatnum2)
    ax.xaxis.set_major_formatter(formatter1)
    ax.yaxis.set_major_formatter(formatter2)
    plt.plot(round, data1[5], label = 'FCUCB-D',color=(0.09019607843137255, 0.7450980392156863, 0.8117647058823529),linewidth=2,marker='v',markersize=9, markevery=1600)
    plt.plot(round, data1[6], label = 'FCTS-D',color=(1.0, 0.4980392156862745, 0.054901960784313725),linewidth=2,marker='s', markersize=9, markevery=1600)
    plt.plot(round, data1[7], label = 'CUCB-D',color=(0.8392156862745098, 0.15294117647058825, 0.1568627450980392),linewidth=2,marker='p', markersize=9, markevery=1600)
    plt.plot(round, data1[8], label = 'MP-TS-D',color=(0.5803921568627451, 0.403921568627451, 0.7411764705882353),linewidth=2,marker='h',markersize=9, markevery=1600)
    plt.plot(round, data1[9], label = 'FGreedy-D',color='goldenrod',linewidth=2,marker='.', markersize=15, markevery=1600)
    plt.tick_params(labelsize=Labelsize)
    plt.grid(alpha=0.2)
    plt.ylabel("Reward regret",fontsize=Fontsize)
    plt.xlim(0,None)
    plt.ylim(0,50)
    plt.xticks(np.arange(0, 20001, 5000))
    plt.legend(fontsize=Legendsize,loc = 'best',framealpha=0.4)
    plt.subplot(2,1,2)
    ax=plt.gca()
    def formatnum1(x, pos):
        return"{:g}".format(x/10000)

    def formatnum2(x, pos):
        return "{:g}".format(x/1000)
    plt.annotate(r'$\times 10^{4}$',(20000,0),(18800,2800*(150/5900)),fontsize = notationsize)
    plt.annotate(r'$\times 10^{3}$',(0,2100),(-5,2800*(5400/5900)),fontsize = notationsize)
    formatter1 = FuncFormatter(formatnum1)
    formatter2 = FuncFormatter(formatnum2)
    ax.xaxis.set_major_formatter(formatter1)
    ax.yaxis.set_major_formatter(formatter2)
    plt.plot(round, data2[5], label = 'FCUCB-D',color=(0.09019607843137255, 0.7450980392156863, 0.8117647058823529),linewidth=2,marker='v',markersize=9, markevery=1600)
    plt.plot(round, data2[6], label = 'FCTS-D',color=(1.0, 0.4980392156862745, 0.054901960784313725),linewidth=2,marker='s', markersize=9, markevery=1600)
    plt.plot(round, data2[7], label = 'CUCB-D',color=(0.8392156862745098, 0.15294117647058825, 0.1568627450980392),linewidth=2,marker='p', markersize=9, markevery=750)
    plt.plot(round, data2[8], label = 'MP-TS-D',color=(0.5803921568627451, 0.403921568627451, 0.7411764705882353),linewidth=2,marker='h',markersize=9, markevery=750)
    plt.plot(round, data2[9], label = 'FGreedy-D',color='goldenrod',linewidth=2,marker='.',markersize=15, markevery=1600)
    plt.tick_params(labelsize=Labelsize)
    plt.grid(alpha=0.2)
    plt.legend(fontsize=Legendsize,loc = 'upper left',framealpha=0.4,bbox_to_anchor=(0, 0.9))
    plt.xlabel("Round",fontsize=Fontsize)
    plt.ylabel("Fairness regret",fontsize=Fontsize)
    plt.xlim(0,None)
    plt.ylim(0,2800)
    plt.xticks(np.arange(0, 20001, 5000))
    plt.tight_layout()
    fig.align_ylabels()
    plt.show()

    fig = plt.figure(figsize=(8,12))
    plt.subplot(2,1,1)
    ax=plt.gca()
    def formatnum1(x, pos):
        return"{:g}".format(x/10000)

    def formatnum2(x, pos):
        return "{:g}".format(x/10)
    plt.annotate(r'$\times 10^{4}$',(20000,0),(18800,60*(150/5900)),fontsize = notationsize)
    plt.annotate(r'$\times 10^{1}$',(0,40),(-5,60*(5400/5900)),fontsize = notationsize)
    formatter1 = FuncFormatter(formatnum1)
    formatter2 = FuncFormatter(formatnum2)
    ax.xaxis.set_major_formatter(formatter1)
    ax.yaxis.set_major_formatter(formatter2)
    plt.plot(round, data1[10], label = 'FCUCB-D',color=(0.09019607843137255, 0.7450980392156863, 0.8117647058823529),linewidth=2,marker='v',markersize=9, markevery=1600)
    plt.plot(round, data1[11], label = 'FCTS-D',color=(1.0, 0.4980392156862745, 0.054901960784313725),linewidth=2,marker='s', markersize=9, markevery=1600)
    plt.plot(round, data1[12], label = 'CUCB-D',color=(0.8392156862745098, 0.15294117647058825, 0.1568627450980392),linewidth=2,marker='p', markersize=9, markevery=1600)
    plt.plot(round, data1[13], label = 'MP-TS-D',color=(0.5803921568627451, 0.403921568627451, 0.7411764705882353),linewidth=2,marker='h',markersize=9, markevery=1600)
    plt.plot(round, data1[14], label = 'FGreedy-D',color='goldenrod',linewidth=2,marker='.', markersize=15, markevery=1600)
    plt.tick_params(labelsize=Labelsize)
    plt.grid(alpha=0.2)
    plt.legend(fontsize=Legendsize,loc = 'best',framealpha=0.4)
    plt.ylabel("Reward regret",fontsize=Fontsize)
    plt.yticks(np.arange(-600, 201, 200))
    plt.xlim(0,None)
    plt.ylim(0,60)
    plt.xticks(np.arange(0, 20001, 5000))
    
    plt.subplot(2,1,2)
    ax=plt.gca()
    def formatnum1(x, pos):
        return"{:g}".format(x/10000)

    def formatnum2(x, pos):
        return "{:g}".format(x/1000)
    plt.annotate(r'$\times 10^{4}$',(20000,0),(18800,4800*(150/5900)),fontsize = notationsize)
    plt.annotate(r'$\times 10^{3}$',(0,4100),(-5,4800*(5400/5900)),fontsize = notationsize)
    formatter1 = FuncFormatter(formatnum1)
    formatter2 = FuncFormatter(formatnum2)
    ax.xaxis.set_major_formatter(formatter1)
    ax.yaxis.set_major_formatter(formatter2)
    plt.plot(round, data2[10], label = 'FCUCB-D',color=(0.09019607843137255, 0.7450980392156863, 0.8117647058823529),linewidth=2,marker='v',markersize=9, markevery=1600)
    plt.plot(round, data2[11], label = 'FCTS-D',color=(1.0, 0.4980392156862745, 0.054901960784313725),linewidth=2,marker='s', markersize=9, markevery=1600)
    plt.plot(round, data2[12], label = 'CUCB-D',color=(0.8392156862745098, 0.15294117647058825, 0.1568627450980392),linewidth=2,marker='p', markersize=9, markevery=750)
    plt.plot(round, data2[13], label = 'MP-TS-D',color=(0.5803921568627451, 0.403921568627451, 0.7411764705882353),linewidth=2,marker='h',markersize=9, markevery=750)
    plt.plot(round, data2[14], label = 'FGreedy-D',color='goldenrod',linewidth=2,marker='.',markersize=15, markevery=1600)
    plt.tick_params(labelsize=Labelsize)
    plt.grid(alpha=0.2)
    plt.legend(fontsize=Legendsize,loc = 'upper left',framealpha=0.4,bbox_to_anchor=(0, 0.9))
    plt.xlabel("Round",fontsize=Fontsize)
    plt.ylabel("Fairness regret",fontsize=Fontsize)
    plt.xlim(0,None)
    plt.ylim(0,4800)
    plt.xticks(np.arange(0, 20001, 5000))
    plt.tight_layout()
    fig.align_ylabels()
    plt.show()


    fig = plt.figure(figsize=(8,12))
    plt.subplot(2,1,1)
    ax=plt.gca()
    def formatnum1(x, pos):
        return"{:g}".format(x/10000)

    def formatnum2(x, pos):
        return "{:g}".format(x/10)
    plt.annotate(r'$\times 10^{4}$',(20000,0),(18800,60*(150/5900)),fontsize = notationsize)
    plt.annotate(r'$\times 10^{1}$',(0,40),(-5,60*(5400/5900)),fontsize = notationsize)
    formatter1 = FuncFormatter(formatnum1)
    formatter2 = FuncFormatter(formatnum2)
    ax.xaxis.set_major_formatter(formatter1)
    ax.yaxis.set_major_formatter(formatter2)
    plt.plot(round, data1[15], label = 'FCUCB-D',color=(0.09019607843137255, 0.7450980392156863, 0.8117647058823529),linewidth=2,marker='v',markersize=9, markevery=1600)
    plt.plot(round, data1[16], label = 'FCTS-D',color=(1.0, 0.4980392156862745, 0.054901960784313725),linewidth=2,marker='s', markersize=9, markevery=1600)
    plt.plot(round, data1[17], label = 'CUCB-D',color=(0.8392156862745098, 0.15294117647058825, 0.1568627450980392),linewidth=2,marker='p', markersize=9, markevery=1600)
    plt.plot(round, data1[18], label = 'MP-TS-D',color=(0.5803921568627451, 0.403921568627451, 0.7411764705882353),linewidth=2,marker='h',markersize=9, markevery=1600)
    plt.plot(round, data1[19], label = 'FGreedy-D',color='goldenrod',linewidth=2,marker='.', markersize=15, markevery=1600)
    plt.tick_params(labelsize=Labelsize)
    plt.grid(alpha=0.2)
    plt.legend(fontsize=Legendsize,loc = 'best',framealpha=0.4)
    plt.ylabel("Reward regret",fontsize=Fontsize)
    plt.xlim(0,None)
    plt.ylim(0,60)
    plt.xticks(np.arange(0, 20001, 5000))
    plt.yticks(np.arange(-900, 201, 300))
    
    plt.subplot(2,1,2)
    ax=plt.gca()
    def formatnum1(x, pos):
        return"{:g}".format(x/10000)

    def formatnum2(x, pos):
        return "{:g}".format(x/1000)
    plt.annotate(r'$\times 10^{4}$',(20000,0),(18800,6200*(150/5900)),fontsize = notationsize)
    plt.annotate(r'$\times 10^{3}$',(0,4100),(-5,6200*(5400/5900)),fontsize = notationsize)
    formatter1 = FuncFormatter(formatnum1)
    formatter2 = FuncFormatter(formatnum2)
    ax.xaxis.set_major_formatter(formatter1)
    ax.yaxis.set_major_formatter(formatter2)
    plt.plot(round, data2[15], label = 'FCUCB-D',color=(0.09019607843137255, 0.7450980392156863, 0.8117647058823529),linewidth=2,marker='v',markersize=9, markevery=1600)
    plt.plot(round, data2[16], label = 'FCTS-D',color=(1.0, 0.4980392156862745, 0.054901960784313725),linewidth=2,marker='s', markersize=9, markevery=1600)
    plt.plot(round, data2[17], label = 'CUCB-D',color=(0.8392156862745098, 0.15294117647058825, 0.1568627450980392),linewidth=2,marker='p', markersize=9, markevery=750)
    plt.plot(round, data2[18], label = 'MP-TS-D',color=(0.5803921568627451, 0.403921568627451, 0.7411764705882353),linewidth=2,marker='h',markersize=9, markevery=750)
    plt.plot(round, data2[19], label = 'FGreedy-D',color='goldenrod',linewidth=2,marker='.',markersize=15, markevery=1600)
    plt.tick_params(labelsize=Labelsize)
    plt.grid(alpha=0.2)
    plt.legend(fontsize=Legendsize,loc = 'upper left',framealpha=0.4,bbox_to_anchor=(0, 0.9))
    plt.xlabel("Round",fontsize=Fontsize)
    plt.ylabel("Fairness regret",fontsize=Fontsize)
    plt.xlim(0,None)
    plt.ylim(0,6200)
    plt.xticks(np.arange(0, 20001, 5000))
    plt.tight_layout()
    fig.align_ylabels()
    plt.show()