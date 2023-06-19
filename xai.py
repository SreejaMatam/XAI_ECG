###################### Analysis with innvestigate
import numpy as np 
import innvestigate
import innvestigate.utils as iutils

def analyze(x, model, bn_model, method, neuron):
  if method == "eps":
    method = "lrp.epsilon"
  if method == "ab0":
    method = "lrp.alpha_1_beta_0"
  if method == "wsq":
    method = "lrp.w_square"
  if method == "psa":
    method = "lrp.sequential_preset_a"
  if method == "igr":
    method = "integrated_gradients"
  relevances = np.empty((len(x),4096,12))
  relevances_l = np.empty((len(x),4096,12))
  analyzer = innvestigate.create_analyzer(method, model, neuron_selection_mode='index')
  analyzer_l = innvestigate.create_analyzer(method, bn_model, neuron_selection_mode='index')
  for aidx, dataset in enumerate(x):
    a = analyzer.analyze(np.expand_dims(dataset, axis=0), neuron_selection=neuron)
    relevances[aidx] = a[0]
    a2 = analyzer_l.analyze(np.expand_dims(dataset, axis=0), neuron_selection=neuron)
    relevances_l[aidx] = a2[0]
  return relevances, relevances_l