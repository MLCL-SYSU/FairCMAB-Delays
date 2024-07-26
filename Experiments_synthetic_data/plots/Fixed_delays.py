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

def Simulation_Fixed_delays():
    data1 = np.loadtxt('./data/Cum_RR_last_round_Fixed_delays.txt')
    data2 = np.loadtxt('./data/Cum_FR_last_round_Fixed_delays.txt')
    FCUCB_D_RR = [data1[0], data1[2], data1[4], data1[6], data1[8], data1[10], data1[12]]
    FCUCB_D_FR = [data2[0], data2[2], data2[4], data2[6], data2[8], data2[10], data2[12]]
    FCTS_D_RR = [data1[1], data1[3], data1[5], data1[7], data1[9], data1[11], data1[13]]
    FCTS_D_FR = [data2[1], data2[3], data2[5], data2[7], data2[9], data2[11], data2[13]]
    delays = np.array([0,50,100,150,200,250,300])
    fig=plt.figure(figsize=(8,8))
    plt.subplot(2,1,1)
    ax=plt.gca()
    def formatnum2(x, pos):
        return '$%d$' % (x/100)
    plt.annotate(r'$\times 10^{2}$',(0,300),(-35,150*(5400/5900)),fontsize = notationsize)
    formatter2 = FuncFormatter(formatnum2)
    ax.yaxis.set_major_formatter(formatter2)

    w=10
    plt.bar(delays-w, FCUCB_D_RR, width=2*w, label = 'FCUCB-D',color=(0.09019607843137255, 0.7450980392156863, 0.8117647058823529),hatch='//')
    plt.bar(delays+w, FCTS_D_RR, width=2*w, label = 'FCTS-D',color=(1.0, 0.4980392156862745, 0.054901960784313725),hatch='\\\\')
    plt.legend(fontsize=Legendsize,loc = 'upper right',framealpha=0.4)
    plt.tick_params(labelsize=Labelsize)
    plt.ylim(0,150)
    plt.ylabel("Reward Regret",fontsize=Fontsize)
    plt.xticks([0,50,100,150,200,250,300])
    ax.axhline(y=0,c='k',ls=':',lw=1) 

    plt.subplot(2,1,2)
    ax=plt.gca()
    def formatnum2(x, pos):
        return '$%d$' % (x/1000)
    plt.annotate(r'$\times 10^{3}$',(0,6800),(-35,6800*(5400/5900)),fontsize = notationsize)
    formatter2 = FuncFormatter(formatnum2)
    ax.yaxis.set_major_formatter(formatter2)

    plt.bar(delays-w, FCUCB_D_FR, width=2*w, label = 'FCUCB-D',color=(0.09019607843137255, 0.7450980392156863, 0.8117647058823529),hatch='//')
    plt.bar(delays+w, FCTS_D_FR, width=2*w, label = 'FCTS-D',color=(1.0, 0.4980392156862745, 0.054901960784313725),hatch='\\\\')
    plt.legend(fontsize=Legendsize,loc = 'upper right',framealpha=0.4)
    plt.xlabel("Delays",fontsize=Fontsize)
    plt.xticks([0,50,100,150,200,250,300])
    plt.tick_params(labelsize=Labelsize)
    plt.ylim(0,6800)
    plt.ylabel("Fairness Regret",fontsize=Fontsize)
    plt.tight_layout()
    fig.align_ylabels()
    plt.show()