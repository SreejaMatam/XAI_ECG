import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy.stats import entropy
from scipy import signal
import pandas as pd
from math import e
import antropy as ant
import pyCompare
from rpy2.robjects import FloatVector
from rpy2.robjects.packages import importr
stats = importr('stats')

###################### Plot Preparations

# prepare scatterplot coloring
cmap = mcolors.ListedColormap(['darkblue','mediumblue','blue','cornflowerblue','lightgrey','orangered','red','crimson','darkred'])
cbar_norm = mcolors.Normalize(vmin=-1, vmax=1)
ax_ylabel = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
ax_ylabel_int = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
abnorms = np.array(["1dAVb","RBBB","LBBB","SB","AF","ST"])
plt.rcParams.update({'font.size': 15})
# create x values for scatter plot
x_range = np.arange(0, 4096, 1)

###################### Plot of Relevances (Scatter)

# plot analysis results as colors on ecg curve. activation as string, relevances only for patient i
def plotRelevances12(x, activation, relevances, method, pid):
    length = len(x)
    size = length/4096
    x_range = np.arange(0, length, 1)
    fig, axs = plt.subplots(1,1,figsize = (int(30*size), 10))
    fig.suptitle("Patient " + pid + "\nMethod: " + method)
    for idx in range(12):
        plt.scatter(x_range, x[:, idx]-idx*2, c = relevances[:, idx], cmap=cm.coolwarm, vmin=-1, vmax=1, s = 15) # linear
    labels = np.arange(0, length/400, 1)
    plt.xticks(np.arange(0, length, 400), [str(round(float(label), 2)) for label in labels])
    plt.yticks(np.arange(0, -24, -2), ax_ylabel)
    plt.xlabel("seconds")
    plt.xlim(0,length)
    plt.ylim(-22.5,1.5)
    fig.savefig('./img/' + str(pid) + '_' + method + '_' + activation + '.pdf', bbox_inches='tight')   # save the figure to file
    plt.close(fig)    # close the figure window

# relevances normed per Lead
def plotRelevances12Normed(x, activation, relevances, method, pid):
    length = len(x)
    size = length/4096
    x_range = np.arange(0, length, 1)
    fig, axs = plt.subplots(1,1,figsize = (int(30*size), 10))
    fig.suptitle("Patient " + pid + "\nMethod: " + method)
    for idx in range(12):
        plt.scatter(x_range, x[:, idx]-idx*2, c = relevances[:, idx]/np.max(np.abs(relevances[:,idx])), cmap=cm.coolwarm, vmin=-1, vmax=1, s = 15) # linear
    labels = np.arange(0, length/400, 1)
    plt.xticks(np.arange(0, length, 400), [str(round(float(label), 2)) for label in labels])
    plt.yticks(np.arange(0, -24, -2), ax_ylabel)
    plt.xlabel("seconds")
    plt.xlim(0,length)
    plt.ylim(-22.5,1.5)
    fig.savefig('./img/' + str(pid) + '_' + method + '_' + activation + '.pdf', bbox_inches='tight')   # save the figure to file
    plt.close(fig)    # close the figure window

# plot analysis results as colors on ecg curve. activation as string, relevances only for patient i
def plotXAIMethodComparison(x, activation, relevances, lead, methods, pid):
    lines = len(methods)
    x_range = np.arange(0, 2001, 1)
    fig, axs = plt.subplots(1,1,figsize = (15, 7))
    fig.suptitle("Patient " + pid)
    for idx in range(lines):
        plt.scatter(x_range, x[:2001, lead]-idx*2, c = relevances[idx][:2001, lead], cmap=cm.coolwarm, vmin=-1, vmax=1, s = 15)
    labels = np.arange(0, 2001/400, 1)
    plt.xticks(np.arange(0, 2001, 400), [str(round(float(label), 2)) for label in labels])
    plt.yticks(np.arange(0, (-2*lines+1), -2), [method.upper() for method in methods])
    plt.xlabel("seconds")
    plt.xlim(0,2001)
    plt.ylim(-2*lines+1,1.5)
    fig.savefig('./img/methodComparison_' + ax_ylabel[lead] + '_' + str(pid) + '_' + activation + '.pdf', bbox_inches='tight')   # save the figure to file
    plt.close(fig)    # close the figure window

# plot analysis results. activation as string, relevances only for patient i
def plotRelevances(x, activation, relevances, method, pid):
    fig, axs = plt.subplots(12,1,figsize = (20, 10))
    fig.suptitle("Patient " + pid + "\nMethod: " + method)
    for idx in range(12):
        plt.subplot(12,1,idx+1)
        plt.plot(x_range, x[:, idx], color="grey", alpha=0.5)
        plt.plot(x_range, relevances[:, idx])
        plt.ylabel(ax_ylabel[idx] + " (μV)", rotation=0, labelpad=30)
        plt.ylim(-1,1)
        if(idx < 11): axs[idx].set_xticks([]) # TODO, not working
    fig.align_ylabels()
    fig.savefig('./img/' + str(pid) + '_' + method + '_' + activation + '.pdf', bbox_inches='tight')   # save the figure to file
    plt.close(fig)    # close the figure window

##################### Plot of Relevances per Lead (Boxplot)
def set_box_color(bp, color):
  plt.setp(bp['boxes'], color=color)
  plt.setp(bp['whiskers'], color=color)
  plt.setp(bp['caps'], color=color)
  plt.setp(bp['medians'], color=color)
  plt.setp(bp['fliers'], markeredgecolor=color)

# plot analysis results per lead as boxplot. relevances for two cohorts as arrays, name will be included in filename.
def plotRelevanceBoxplot(relevancesAF, relevancesNormal, nameGroup1, nameGroup2, name, method):
  xAF = np.mean(relevancesAF, axis=1)
  xNormal = np.mean(relevancesNormal, axis=1)
  color_abnorm = '#386cb0'
  if nameGroup1 == "AF": color_abnorm = '#267326'
  # init label
  lead_labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
  # iterate over every lead
  for i in range(12):
    # perform one sided wilcoxon rank sum test
    test = stats.wilcox_test(FloatVector(xAF[:, i]), FloatVector(xNormal[:, i]), alternative='greater', paired=False)
    d = {key: test.rx2(key)[0] for key in ['statistic', 'p.value', 'alternative']}
    p_val = d["p.value"]
    # perform an unpaired t-test
    #d = scipy.stats.ttest_ind(xAF[:, i], xNormal[:, i], equal_var=False)
    #p_val = d[i]
    # calculate difference in means
    #diff = np.mean(xAF[:, i]) - np.mean(xNormal[:, i])
    #diff = round(diff, 1)
#    print(p_val)
    # if the ttest is significant (p-value < 0.01) append an Asterix (*) to the Lead Label and add mean difference in next row
    if p_val < 0.0001:
      lead_labels[i] = str(lead_labels[i]) + "*\n " #+ str(diff)
    else:
      lead_labels[i] = str(lead_labels[i]) + "\n " #+ str(diff)
  fig = plt.figure(figsize = (6, 3))
  ax1 = fig.add_subplot(111)
  ticks = lead_labels
  bpl = plt.boxplot(xAF, positions=np.array(range(len(ticks)))*2.0-0.4, widths=0.6)
  bpr = plt.boxplot(xNormal, positions=np.array(range(len(ticks)))*2.0+0.4, widths=0.6)
  set_box_color(bpl, color_abnorm) # colors are from http://colorbrewer2.org/
  set_box_color(bpr, 'gray')
  # draw temporary red and blue lines and use them to create a legend
  ax1.plot([], c=color_abnorm, label=nameGroup1)
  ax1.plot([], c='gray', label=nameGroup2)
  ax1.legend(prop={'size': 10})
  plt.xticks(range(0, len(ticks) * 2, 2), ticks, fontsize=10)
  plt.xlim(-2, len(ticks)*2)
  #plt.yticks([-5,-2.5,0,2.5,5], fontsize=12)
  #plt.ylim([-5, 5])  
  ax1.set_xticklabels(lead_labels)  
  #plt.ylim(-0.1, 1)
  #plt.xlabel("ECG Lead, Significance, Difference of means")
  plt.grid(True)
  plt.xlabel("$k$")
  plt.ylabel("Mean of Relevances $M_{n,k}$", fontsize=12)
  ax2 = ax1.twiny()
  plt.xticks(range(0, len(ticks) * 2, 2), ticks, fontsize=12)
  plt.xlim(-2, len(ticks)*2)
  #plt.yticks([-5,-2.5,0,2.5,5], fontsize=12)
  #plt.ylim([-5, 5])
  ax2.set_xticklabels(ax_ylabel)  
  plt.tight_layout()
  plt.savefig('./img/boxplotMean_' + nameGroup1 + 'vs' + nameGroup2 + '_' + name + '_' + method + '.pdf', bbox_inches='tight')


def plotEntropyBoxplot(relevancesAF, relevancesNormal, nameGroup1, nameGroup2, name, method):
  xAF = []
  xNormal = []
  color_abnorm = '#386cb0'
  if nameGroup1 == "AF": color_abnorm = '#267326'
  for i in range(len(relevancesAF)):
    patient = []
    v1_entropy = 0
    for j in range(12):
      a = np.swapaxes(relevancesAF[i],0,1)[j]
      a_result = ant.sample_entropy(signal.detrend(a))
      if j==6:
        v1_entropy = a_result
      patient.append(a_result)
    xAF.append(patient)
  xAF = np.array(xAF)
  for i in range(len(relevancesNormal)):
    patient = []
    v1_entropy = 0
    for j in range(12):
      b = np.swapaxes(relevancesNormal[i],0,1)[j]
      b_result = ant.sample_entropy(signal.detrend(b))
      if j==6:
        v1_entropy = b_result
      patient.append(b_result)
    xNormal.append(patient)
  xNormal = np.array(xNormal)
  # init label
  lead_labels = [0 for x in range(12)]
  # plot
  fig = plt.figure(figsize = (6, 3))
  ax1 = fig.add_subplot(111)
  ticks = lead_labels
  bpl = plt.boxplot(xAF, positions=np.array(range(len(ticks)))*2.0-0.4, widths=0.6)
  bpr = plt.boxplot(xNormal, positions=np.array(range(len(ticks)))*2.0+0.4, widths=0.6)
  set_box_color(bpl, color_abnorm) # colors are from http://colorbrewer2.org/
  set_box_color(bpr, 'gray')
  # draw temporary lines and use them to create a legend
  ax1.plot([], c=color_abnorm, label=nameGroup1)
  ax1.plot([], c='gray', label=nameGroup2)
  ax1.legend(prop={'size': 10})
  plt.xticks(range(0, len(ticks) * 2, 2), ticks, fontsize=10)
  plt.xlim(-2, len(ticks)*2)
  ax1.set_xticklabels(ax_ylabel_int)
  plt.grid(True)
  plt.xlabel("$k$")
  plt.ylabel("Shannon Entropy", fontsize=12)
  ax2 = ax1.twiny()
  plt.xticks(range(0, len(ticks) * 2, 2), ticks, fontsize=12)
  plt.xlim(-2, len(ticks)*2)
  ax2.set_xticklabels(ax_ylabel)  
  plt.tight_layout()
  plt.savefig('boxplot_entropy_' + nameGroup1 + 'vs' + nameGroup2 + '_' + name + '.pdf', bbox_inches='tight')

##################### Plot of Relevances (Heatmap)

# plot heatmap comparing activations. relevances only for patient i
def activationHeatmap(relevancesS, relevancesL, method, pid):
  fig, (ax1, ax2) = plt.subplots(2,1,figsize = (20, 5))
  fig.suptitle("Patient " + str(pid) + ", Method: " + str(method), fontsize=16)
  ax1.set_title('sigmoid activation') 
  ax1.imshow(relevancesS.squeeze().T, cmap=cmap, vmin=-1, vmax=1, interpolation='nearest', aspect='auto')
  ax2.set_title('linear activation') 
  ax2.imshow(relevancesL.squeeze().T, cmap=cmap, vmin=-1, vmax=1, interpolation='nearest', aspect='auto')
  fig.tight_layout()
  fig.savefig('./img/Heatmap_' + str(pid) + '_' + method + '.pdf')   # save the figure to file
  plt.close(fig) 
###################### Plot of Classes

# plot analysis results as colors on ecg curve. activation as string, relevances only for patient i
def plotClassBoxplot(label_abnorm, label_sinus, name1, name2):
  plt.figure(figsize = (8, 4))
  ticks = abnorms
  bpl = plt.boxplot(label_abnorm, positions=np.array(range(len(ticks)))*2.0-0.4, sym='', widths=0.6)
  bpr = plt.boxplot(label_sinus, positions=np.array(range(len(ticks)))*2.0+0.4, sym='', widths=0.6)
  set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
  set_box_color(bpr, '#2C7BB6')
  # draw temporary red and blue lines and use them to create a legend
  plt.plot([], c='#D7191C', label=name1)
  plt.plot([], c='#2C7BB6', label=name2)
  plt.legend()
  plt.xticks(range(0, len(ticks) * 2, 2), ticks)
  plt.xlim(-2, len(ticks)*2)
  plt.ylim(-0.1, 1)
  plt.xlabel("DNN-class (ECG Abnormality)")
  plt.ylabel("Class probability")
  plt.grid(True)
  plt.tight_layout()
  plt.savefig('./img/boxcompare_' + name1 + 'vs' + name2 + '.pdf', bbox_inches='tight')

###################### Comparison of Activations

# plot difference between normed sigmoid and linear values
def compareActivations(norm, norm_l, method):
  diff = norm - norm_l
  fig, ax = plt.subplots(len(x),1,figsize = (20, 10))
  fig.suptitle("Method: " + str(method), fontsize=16)
  for idx in range(len(x)):
    ax[idx].set_title("Patient " + str(pid[idx]))
    ax[idx].scatter(x_range, diff[idx][:, 0], s = 2) # linear
  fig.tight_layout()
  plt.show()
  # Bland-Altman Plot
  fig, ax = plt.subplots(len(x),12,figsize = (25, 15))
  fig.suptitle("Sigmoid vs. Linear Activation, " + str(method), fontsize=16)
  for idx in range(len(x)):
    for lead in range(12):
      ax[idx,lead] = pyCompare.blandAltman(norm[idx][:,lead], norm_l[idx][:,lead], ax=ax[idx,lead])
  fig.tight_layout()
  plt.savefig('./img/blandaltman_' + str(method) + '.pdf', bbox_inches='tight')

