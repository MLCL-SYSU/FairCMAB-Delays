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


def Simulation_Geometric_delays():
    data1 = np.loadtxt('./data/Cum_RR_Geometric_delays.txt')
    data2 = np.loadtxt('./data/Cum_FR_Geometric_delays.txt')
    data3 = np.loadtxt('./data/Selection_pro_Geometric_delays.txt')
    
    round = np.array(list(range(len(data1[0]))))
    arm_list = np.array(list(range(len(data3[0]))))+1
    
    # plot average selection fractions
    plt.figure(figsize=(8.5,8))
    w=0.14
    plt.bar(arm_list-2.5*w, data3[0], width=w, label = 'FCUCB-D',color=(0.09019607843137255, 0.7450980392156863, 0.8117647058823529),hatch='//')
    plt.bar(arm_list-1.5*w, data3[1], width=w, label = 'FCTS-D',color=(1.0, 0.4980392156862745, 0.054901960784313725),hatch='\\\\')
    plt.bar(arm_list-0.5*w, data3[4], width=w, label = 'FGreedy-D',color='goldenrod',hatch='xx')
    plt.bar(arm_list+0.5*w, data3[2], width=w, label = 'CUCB-D',color=(0.8392156862745098, 0.15294117647058825, 0.1568627450980392),hatch='++')
    plt.bar(arm_list+1.5*w, data3[3], width=w, label = 'MP-TS-D',color=(0.5803921568627451, 0.403921568627451, 0.7411764705882353),hatch='--')
    plt.bar(arm_list+2.5*w, data3[5], width=w, label = 'Optimal',color='grey')
    plt.xticks(np.arange(1, len(data3[0])+1, 1))
    plt.legend(fontsize=Legendsize,loc = 'upper left',framealpha=0.4)
    plt.xlabel("Arms",fontsize=Fontsize)
    plt.tick_params(labelsize=Labelsize)
    plt.ylabel("Arm Selection Fractions",fontsize=Fontsize)
    plt.tight_layout()
    plt.show()

    # plot reward regret and fairnes regret
    fig = plt.figure(figsize=(8.5,8))
    plt.subplot(2,1,1)

    ax=plt.gca()
    def formatnum1(x, pos):
        return"{:g}".format(x/10000)
        # return x/10000
    def formatnum2(x, pos):
        return"{:g}".format(x/10)
    plt.annotate(r'$\times 10^{4}$',(20000,0),(18900,50*(150/5900)),fontsize = notationsize)
    plt.annotate(r'$\times 10^{1}$',(0,40),(-5,50*(150/5900)),fontsize = notationsize)
    formatter1 = FuncFormatter(formatnum1)
    formatter2 = FuncFormatter(formatnum2)
    ax.xaxis.set_major_formatter(formatter1)
    ax.yaxis.set_major_formatter(formatter2)


    plt.plot(round, data1[0], label = 'FCUCB-D',color=(0.09019607843137255, 0.7450980392156863, 0.8117647058823529),linewidth=2,marker='v', markersize=9, markevery=1600)
    plt.plot(round, data1[1], label = 'FCTS-D',color=(1.0, 0.4980392156862745, 0.054901960784313725),linewidth=2,marker='s',markersize=9, markevery=1600)
    plt.plot(round, data1[4], label = 'FGreedy-D',color='goldenrod',linewidth=2,marker='.', markersize=15, markevery=1600)
    plt.plot(round, data1[2], label = 'CUCB-D',color=(0.8392156862745098, 0.15294117647058825, 0.1568627450980392),linewidth=2,marker='p', markersize=9, markevery=725)
    plt.plot(round, data1[3], label = 'MP-TS-D',color=(0.5803921568627451, 0.403921568627451, 0.7411764705882353),linewidth=2,marker='h',markersize=9, markevery=725)
    plt.ylim(0,50)
    plt.xlim(0,None)
    plt.xticks(np.arange(0, 20001, 5000))
    plt.yticks(np.arange(-600, 150, 200))
    plt.tick_params(labelsize=Labelsize)
    plt.grid(alpha=0.2)
    plt.legend(fontsize=Legendsize,loc = 'lower center',framealpha=0.4,ncol=2,columnspacing=0.5,bbox_to_anchor=(0.6, -0.03))
    plt.ylabel("Reward Regret",fontsize=Fontsize)
    plt.tight_layout()


    plt.subplot(2,1,2)

    ax=plt.gca()
    def formatnum1(x, pos):
        return"{:g}".format(x/10000)
        # return x/10000
    def formatnum2(x, pos):
        return "{:g}".format(x/1000)
    plt.annotate(r'$\times 10^{4}$',(20000,0),(18900,2800*(150/5900)),fontsize = notationsize)
    plt.annotate(r'$\times 10^{3}$',(0,100),(-5,2800*(150/5900)),fontsize = notationsize)
    formatter1 = FuncFormatter(formatnum1)
    formatter2 = FuncFormatter(formatnum2)
    ax.xaxis.set_major_formatter(formatter1)
    ax.yaxis.set_major_formatter(formatter2)


    plt.plot(round, data2[0], label = 'FCUCB-D',color=(0.09019607843137255, 0.7450980392156863, 0.8117647058823529),linewidth=2,marker='v', markersize=9, markevery=1600)
    plt.plot(round, data2[1], label = 'FCTS-D',color=(1.0, 0.4980392156862745, 0.054901960784313725),linewidth=2,marker='s',markersize=9, markevery=1600)
    plt.plot(round, data2[4], label = 'FGreedy-D',color='goldenrod',linewidth=2,marker='.',markersize=15, markevery=1600)
    plt.plot(round, data2[2], label = 'CUCB-D',color=(0.8392156862745098, 0.15294117647058825, 0.1568627450980392),linewidth=2,marker='p', markersize=9, markevery=750)
    plt.plot(round, data2[3], label = 'MP-TS-D',color=(0.5803921568627451, 0.403921568627451, 0.7411764705882353),linewidth=2,marker='h',markersize=9, markevery=750)
    plt.ylim(0,2800)
    plt.xticks(np.arange(0, 20001, 5000))
    plt.tick_params(labelsize=Labelsize)
    plt.grid(alpha=0.2)
    plt.legend(fontsize=Legendsize,loc = 'lower center',framealpha=0.4,ncol=2,columnspacing=0.5,bbox_to_anchor=(0.6, 0.51))
    plt.xlabel("Round",fontsize=Fontsize)
    plt.ylabel("Fairness Regret",fontsize=Fontsize)
    plt.xlim(0,None)
    plt.tight_layout()
    fig.align_ylabels()
    plt.show()