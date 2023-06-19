# test environment: Python 3.6.8, tensorflow 1.12.0 (Windows10)
import os
import re
import numpy as np
import scipy.io
import ecgdata
import ribeiro
import xai
#import visualization as viz


input_dir = "./data/"
dataset = "ptbxl" # options: umg, ribeiro, cpsc, ptbxl
batchsize = 1 # for ribeiro model
# set LRP method or Integrated Gradient
method = "igr" # options: Epsilon - eps,  Alpha1Beta0 - ab0, WSquare - wsq, PresetA - psa, IGradient - igr
neuron = 1 # 1=1AVB, 2=LBBB, 4=AF
cohortsize = 200
#matfile = 'PTB_1AVB.mat'
matfile = 'PTB_1AVB_NORM_0000-1000.mat'

###################### Choose patients

### CPSC2018
#idx_AFfp = np.array([4096, 5632, 3076, 6661, 4616, 5642, 2571, 3595, 1551, 4629, 3605, 5143, 6175, 547, 3620, 1063, 6184, 2089, 1066, 1579, 3627, 4653, 6190, 3118, 1073, 4666, 2619, 2623, 1099, 6732, 1100, 604, 6757, 6760, 6762, 6255, 5232, 5111, 1663, 6784, 641, 3719, 3721, 2698, 2188, 4238, 4759, 5276, 4254, 6815, 160, 5281, 6307, 5796, 2213, 6311, 1195, 4269, 6321, 4786, 3252, 181, 2746, 6331, 701, 5310, 1730, 4806, 201, 3793, 4822, 739, 3299, 2799, 3311, 2289, 6390, 1272, 5884, 2302, 1797, 1806, 3344, 1299, 4387, 292, 2855, 809, 299, 6451, 1844, 2877, 1346, 2380, 4430, 2383, 2900, 1368, 3416, 3930, 2395, 4963, 5992, 6508, 876, 1902, 1391, 880, 4465, 5996, 3445, 4471, 5496, 3451, 2941, 1407, 384, 387, 901, 5509, 5001, 4493, 2446, 2447, 5519, 1427, 2452, 1941, 1437, 1438, 3485, 3487, 6565, 4517, 1962, 4524, 4530, 6066, 4538, 3517, 5567, 2503, 3017, 5066, 1993, 3532, 1486, 4561, 3029, 1494, 2009, 2019, 5604, 2535, 4077, 505, 1520, 3571, 3575, 2552, 3065])
#idx_x = np.array([4096, 5632, 3076, 2, 3, 6]) # 3x FP, 2x TP, 1x FN
#idx_x = np.array([2, 3, 37, 93, 188, 418, 575, 823]) # 2x TP, 6x TN
#idx_x = np.array([100, 221, 389, 518, 940, 988, 1016]) #Relevanzen V1 < 0
#idx_x = np.array([266, 638]) #Relevanzen V1 > 0.4 
#idx_x = np.array([555, 917, 984, 1272, 3370, 4963, 5563, 5701, 5773]) #LBBB Ausreißer
#idx_abnorm, idx_Sinus = ecgdata.getCohortsCPSC(neuron)
#idx_x = np.concatenate([idx_abnorm[:cohortsize], idx_Sinus[:cohortsize]], axis=0)

### Ribeiro 
#idx_x = np.array([170, 355, 368, 120, 259, 485]) # select 3 FN and 3 TP AFs
#idx_x = np.array([170, 355, 368, 120, 259, 348, 408, 415, 485, 501, 548, 564, 572]) # select all AFs

### PTB-XL
idx_x = 'NORM' # superclasses: 1AVB, CLBBB, CRBBB, AFIB, STACH, SBRAD, NORM

x, pid = ecgdata.load(input_dir, dataset, idx_x)

model, bn_model = ribeiro.loadmodel(input_dir)
label, label_linear = ribeiro.getlabels(x, model, bn_model, batchsize)
relevances, relevances_l = xai.analyze(x, model, bn_model, method, neuron)

scipy.io.savemat(matfile, mdict={'raw': x, 'rel': relevances, 'rel_linear': relevances_l, 'confidence': label, 'confidence_linear': label_linear, 'pid': pid})

#scipy.io.savemat("PTB_LBBB_RibeiroLabels", mdict={'raw': x, 'confidence': label, 'confidence_linear': label_linear, 'pid': pid})
