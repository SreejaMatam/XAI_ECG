###################### DL Model Preparation
import keras
import copy

def loadmodel(input_dir):
	# load Ribeiro model - https://zenodo.org/record/3765717
	model = keras.models.load_model(input_dir + "model.hdf5", compile=False)
	for l in model.layers:
	  l.name = "%s_workaround" % l.name
	# Create model with new names
	model = keras.models.Model(input=model.input, output=model.output)
	model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam())
	orig_model = copy.copy(model)
	# replace sigmoid activation with linear (s. https://github.com/keras-team/keras/issues/7190)
	bn_layer = model.get_layer(index=-1)
	bn_layer.activation = keras.activations.get('linear')
	bn_prev_layer = model.get_layer(index=-2)
	bn_output = bn_layer(bn_prev_layer.output)
	bn_model = keras.Model(inputs=model.inputs, outputs=bn_output)
	return orig_model, bn_model
	
def getlabels(x, model, bn_model, batchsize=1):
	# create Tensor for DL model input
	arr_tensor = keras.backend.constant(x)
	if batchsize==1:
		# analyse one batch
		label = keras.backend.get_value(model(arr_tensor))
		label_linear = keras.backend.get_value(bn_model(arr_tensor))
	else:
		# analyse multiple batches
		y = tf.split(arr_tensor, num_or_size_splits=batchsize, axis=0)
		firstiteration = True
		label = []
		label_linear = []
		for i in range(len(y)):   
		  if firstiteration:
		    label = keras.backend.get_value(model(y[i]))
		    label_linear = keras.backend.get_value(bn_model(y[i]))
		    firstiteration = False
		  else:
		    label_i = keras.backend.get_value(model(y[i]))
		    label_linear_i = keras.backend.get_value(bn_model(y[i]))
		    label = np.concatenate([label, label_i], axis = 0)
		    label_linear = np.concatenate([label_linear, label_linear_i], axis = 0)
	return label, label_linear