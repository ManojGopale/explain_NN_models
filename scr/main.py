from utils.newLoadData import manageData
#from utils.classify_general

import numpy as np
import json
import tensorflow as tf
import tensorflow.keras as keras
from keras.models import load_model

## Python 3.9 and 3.10 version imports
import sys
if(sys.version_info > (3,9)):
	import keras.utils as np_utils
else:
	from keras.utils import np_utils

from optparse import OptionParser

tf.compat.v1.disable_eager_execution()

import innvestigate

parser = OptionParser()
parser.add_option('--modelType',
									action = 'store', type='string', dest='modelType', default = 'FNN')
parser.add_option('--chipType',
									action = 'store', type='string', dest='chipType', default = 'cw_f303_1500')
parser.add_option('--totalTraces',
									action = 'store', type='string', dest='totalTraces', default = "15k")

(options, args) = parser.parse_args()

########
modelType = options.modelType
chipType  = options.chipType ##This is the chip data that we want to load
totalTraces = options.totalTraces
#######

#####----Load parameters from file----------#####
filePath="/xdisk/bethard/manojgopale/extra/keyPrediction_chip/scr/moreDataTrials/result/analysis_config.json"
with open(filePath) as f:
	params = json.load(f)

paramsDict = params[chipType][modelType][totalTraces]
dataFolder = paramsDict["dataFolder"]
numPowerTraces = paramsDict["numPowerTraces"]
modelName = paramsDict["modelName"]
resultDir = paramsDict["resultDir"]
datagen = paramsDict["datagen"]
typeOfStd = paramsDict["typeOfStd"]
randInt = paramsDict["randInt"]

#####-----------------------------------------#####

##1. Load dataset
##2. Pre-process data
##3. Load model
##4. Investigate library for checking relevant inputs key-wise

#npzDir = "/xdisk/tosiron/manojgopale/xdisk/gem5DataCollection/compressed_npyDatasets/"
npzDir = "../data/compressed_npyDatasets/"
dataset = manageData()
x_train, y_train, x_dev, y_dev = dataset.loadData(npzDir=npzDir, chipType=chipType,  randInt=randInt, totalTraces=totalTraces)
##y_train and y_dev needs to be numpy for further analysis
y_train = y_train.values
y_dev = y_dev.values

##Pre-process data
x_train, x_dev = dataset.processData(x_train, x_dev, typeOfStd=typeOfStd)

print("Started one hot\n")
n_classes = 256
y_train_oh = np_utils.to_categorical(y_train, n_classes)
y_dev_oh = np_utils.to_categorical(y_dev, n_classes)
print("Done one hot\n")

if(modelType == "CNN"):
	x_train = x_train.reshape(-1,numPowerTraces,1)
	x_dev = x_dev.reshape(-1,numPowerTraces,1)
elif(modelType == "RNN"):
	x_train = x_train.reshape(-1,1,numPowerTraces) ##batchSize, timesteps, features
	x_dev = x_dev.reshape(-1,1,numPowerTraces) ##batchSize, timesteps, features
	#	y_oh = y_oh.reshape(-1,1,1) ##When not using one-hot
	y_train_oh = y_train_oh.reshape(-1,1,256) ##When we use one-hot the shape should be y_data.reshape(-1,1,256)
	y_dev_oh = y_dev_oh.reshape(-1,1,256) ##When we use one-hot the shape should be y_data.reshape(-1,1,256)
elif(modelType=="SCAAML"):
	##Reshape data for SCAAML model
	x_train = x_train.reshape(-1,numPowerTraces,1)
	x_dev = x_dev.reshape(-1,numPowerTraces,1)
	input_shape = x_train.shape[1:]
##########--------------------npz data--------------------###############

#modelPath=resultDir + "/" + chipType + "/" + modelName
modelPath="../models/" + modelName
#modelPath="/xdisk/tosiron/manojgopale/extra/keyPrediction_chip/scr/moreDataTrials/scr/../result/cw_f303_6000/" + modelName
if(modelType == "SCAAML"):
	config = { "batchSize": 16000,  
			"model_parameters": {
			"activation": "relu",
			"initial_filters": 8,
			"initial_pool_size": 4,
			"block_kernel_size": 3,
			"blocks_stack1": 3,
			"blocks_stack2": 4,
			"blocks_stack3": 4,
			"blocks_stack4": 3,
			"dense_dropout": 0.1
		}
	}
	model = classify_general.SCAAML_model().get_model(input_shape, config)
else:
	model = load_model(modelPath)

model.compile(loss='categorical_crossentropy', metrics=['categorical_accuracy'], optimizer='adam')
print(f"Loaded model from {modelPath}")

score_dev=model.evaluate(x_dev,y_dev_oh, batch_size=10240)
print(f"score dev accuracy = {score_dev[1]}")


##https://github.com/albermax/innvestigate/blob/master/src/innvestigate/backend/graph.py#L266
import tensorflow.keras.backend as kbackend
import tensorflow.keras.layers as klayers
import tensorflow.keras.models as kmodels

import innvestigate.backend as ibackend
import innvestigate.backend.checks as ichecks
import innvestigate.layers as ilayers



