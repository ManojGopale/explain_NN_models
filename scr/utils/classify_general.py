import tensorflow as tf
import tensorflow.keras as keras
#from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, LSTM, GRU, Bidirectional, Conv1D, MaxPooling1D, Flatten,AveragePooling1D
from keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint, EarlyStopping
#from keras.layers.normalization import BatchNormalization
from keras.layers import BatchNormalization
import pickle
import gzip
import pandas as pd
import numpy as np
import gc
import os
import multiprocessing, pprint
import time
from itertools import repeat
from multiprocessing import Process, Pool
from scipy.signal import stft
#from tqdm import tqdm
from datetime import date, datetime
from sklearn.utils import shuffle

from sklearn.utils import shuffle
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, scale, maxabs_scale
from sklearn.metrics import confusion_matrix
from keras.models import clone_model
from keras.models import load_model

import glob, os, itertools, pdb, copy, time
import collections, json
from sklearn.model_selection import train_test_split
from collections import Counter
from scipy.spatial import distance


from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objs as go


## Python 3.9 and 3.10 version imports
import sys
if(sys.version_info > (3,9)):
	import keras.utils as np_utils
else:
	from keras.utils import np_utils
##Since the path to error_analysis file is already added to sys in run_3HL.py, we will directly 
##import those here

#from error_analysis import errorAnalysis

## Commented for config5p4 to re-run
np.random.seed(9)

scaler = StandardScaler()

def process_inputs (dataPath):
	data = pd.read_pickle(dataPath)
	## Data is already shuffled during saving
	#dataShuffle = shuffle(data)
	## Apply to get the 1'st element of the entire key
	y_data = data.key.apply(lambda x: x[0]).values
	## For x_data, awe need to first convert the list of memmap array to separate columns of numbers .apply(pd.Series) does that
	## .values converts it to an array
	## then we apply scaler.fir_trasnform to it
	#x_data = scaler.fit_transform(data.trace.apply(pd.Series).values)

	## Without transformation
	x_data = data.trace.apply(pd.Series).values
	return x_data, y_data

## dataPath : Path where the working directory is located. 
## trainSize : Number of power traces per key to take
## testFlag : if true, then traina nd dev data won't be loaded
def getData(dataPath, moreDataPath, trainSize, trainFlag, devFlag, testFlag):
	##TODO changed devSize to 1000 for trails for more trainSize data. It was 5000 for default run
	devSize  = 1000
	testSize = 1000
	numTraces = 1500 ##Number of traces collected per key

	## Pre defining the arrays based on sizes of the data
	x_train = np.zeros((trainSize*256, numTraces))
	x_dev = np.zeros((devSize*256, numTraces))
	x_test = np.zeros((testSize*256, numTraces))

	y_train = np.zeros((trainSize*256, 1))
	y_dev = np.zeros((devSize*256, 1))
	y_test = np.zeros((testSize*256, 1))

	for index, val in enumerate(range(0,256)):
		print("Started data processing for %d key\n" %(val))
		trainStr = dataPath + "train_" + str(val) + ".pkl.zip"
		devStr   = dataPath + "dev_" + str(val) + ".pkl.zip"
		testStr  = dataPath + "test_" + str(val) + ".pkl.zip"

		##more training data path
		moreTrainStr = moreDataPath + "train_" + str(val) + ".pkl.zip"

		## Checking if the file size is 0, before processing data
		## This check is for cross config analysis, where traina nd dev are empty
		#if (os.stat(trainStr).st_size != 0):
		if (trainFlag):
			x_train_inter, y_train_inter = process_inputs(trainStr)
			## Trainsize will still be 15000, but we will take data from devSet to trainset
			x_train[trainSize*index: trainSize*(index) + 15000, : ] = x_train_inter
			y_train[trainSize*index: trainSize*(index) + 15000, 0] = y_train_inter

			if(trainSize == 28000):
				## Adding 9000 more data
				x_train_inter_more, y_train_inter_more = process_inputs(moreTrainStr)
				x_train[trainSize*(index) + 15000: (trainSize*(index) + 15000) + 9000, : ] = x_train_inter_more[0:9000, :]
				y_train[trainSize*(index) + 15000: (trainSize*(index) + 15000) + 9000, 0] = y_train_inter_more.reshape(9000,1)[0:9000, 0]

			print("Train= %s\n" %(trainFlag))
		else: 
			## Assigning the array's to 0's
			##NOTE: needs to change shape, but since we are always training, I am not changing this
			x_train[trainSize*index: trainSize*(index+1), : ] = np.zeros((trainSize,numTraces))
			y_train[trainSize*index: trainSize*(index+1), : ] = np.zeros((trainSize, 1))
			print("train= %s\n" %(trainFlag))

		#if (os.stat(devStr).st_size != 0):
		if (devFlag):
		## get the data for each sub part
			x_dev_inter, y_dev_inter     = process_inputs(devStr)
			print("x_dev_inter= %s, y_dev_inter= %s" %(x_dev_inter.shape, y_dev_inter.shape))
			x_dev[devSize*index: devSize*(index+1), : ] = x_dev_inter[0:devSize, :]
			y_dev[devSize*index: devSize*(index+1), 0 ] = y_dev_inter.reshape(5000, 1)[0:devSize, 0]
			print("Dev= %s\n" %(devFlag))
			print("x_dev= %s, y_dev= %s" %(x_dev.shape, y_dev.shape))

			if(trainSize==28000):
				## Adding 4000 traces to trainSet here
				x_train[trainSize*(index) + 15000 +9000: (trainSize*(index) + 15000) + 13000, : ] = x_dev_inter[1000:5000, :]
				y_train[trainSize*(index) + 15000 +9000: (trainSize*(index) + 15000) + 13000, 0] = y_dev_inter.reshape(5000, 1)[devSize:5000, 0]
				print("x_trainSize = %s, y_trainSize= %s" %(x_train.shape, y_train.shape))
		else:
			x_dev[devSize*index: devSize*(index+1), : ] = np.zeros((devSize, numTraces))
			y_dev[devSize*index: devSize*(index+1), : ] = np.zeros((devSize, 1))
			print("dev= %s\n" %(devFlag))

		## Test data is present so check is not performed
		if (testFlag):
			x_test_inter, y_test_inter   = process_inputs(testStr)
			x_test[testSize*index: testSize*(index+1), : ] = x_test_inter
			y_test[testSize*index: testSize*(index+1), 0 ] = y_test_inter
			print("Test= %s\n" %(testFlag))
			print("x_test= %s, y_test= %s" %(x_test.shape, y_test.shape))
		else:
			x_test[testSize*index: testSize*(index+1), : ] = np.zeros((testSize, numTraces))
			y_test[testSize*index: testSize*(index+1), : ] = np.zeros((testSize, 1))
			print("test= %s\n" %(testFlag))

		print("Finished data processing for %d key\n" %(val))
	
	## Clear variables
	x_train_inter = None
	x_dev_inter = None
	x_test_inter = None
	y_train_inter = None
	y_dev_inter = None
	y_test_inter = None
	x_train_inter_more = None
	y_train_inter_more = None
	print("\nCleared variables\n")

	##Not shuffling for debugging, should be removed
	## Shuffling
	## https://scikit-learn.org/stable/modules/generated/sklearn.utils.shuffle.html
	print("\nStarted shuffling of data\nx_train[0]= %s\ny_train[0]= %s" %(x_train[0], y_train[0]))
	print("\nx_train[12000]= %s\ny_train[12000]= %s" %(x_train[12000], y_train[12000]))
	x_train, y_train = shuffle(x_train, y_train, random_state=0)
	x_dev, y_dev = shuffle(x_dev, y_dev, random_state=0)
	x_test, y_test = shuffle(x_test, y_test, random_state=0)
	print("\nFinished shuffling of data\nx_train[0]= %s\ny_train[0]= %s" %(x_train[0], y_train[0]))
	print("\nx_train[12000]= %s\ny_train[12000]= %s" %(x_train[12000], y_train[12000]))

	##NOTE: Remove:
	#Mimport pdb; pdb.set_trace()
	## One hot assignment
	n_classes = 256
	y_train_oh = np_utils.to_categorical(y_train, n_classes)
	y_dev_oh = np_utils.to_categorical(y_dev, n_classes)
	y_test_oh = np_utils.to_categorical(y_test, n_classes)
	
	print("\nOne-hot encoded for outputs\n")
	## Standardizing train, dev and test
	x_train_mean = x_train.mean(axis=0)
	x_train_std  = x_train.std(axis=0)

	x_dev_mean = x_dev.mean(axis=0)
	x_dev_std = x_dev.mean(axis=0)

	x_test_mean = x_test.mean(axis=0)
	x_test_std = x_test.std(axis=0)

	#M## Concatenating train and dev
	#Mx_full = np.concatenate((x_train, x_dev), axis=0)
	#Mx_full_mean = x_full.mean(axis=0)
	#Mx_full_std = x_full.std(axis=0)

	## chunking the normalization process
	print("Strated normalizing\n")
	chunkSize = 28000
	chunkNum = int(len(x_train)/chunkSize)
	for chunkIndex in range(chunkNum):
		print("Train chunkIndx= %s, chunkNum = %s" %(chunkIndex, chunkNum))
		if(chunkIndex != chunkNum-1): 
			x_train[chunkIndex*chunkSize: (chunkIndex+1)*chunkSize] = (x_train[chunkIndex*chunkSize: (chunkIndex+1)*chunkSize]-x_train_mean)/x_train_std
		else:
			x_train[chunkIndex*chunkSize: ] = (x_train[chunkIndex*chunkSize: ] - x_train_mean)/x_train_std
			
	devChunkSize = 10000
	devChunkNum = int(len(x_dev)/devChunkSize)
	for devChunkIndex in range(devChunkNum):
		print("Dev chunkIndx= %s, chunkNum = %s" %(devChunkIndex, devChunkNum))
		if(devChunkIndex != devChunkNum-1): 
			x_dev[devChunkIndex*devChunkSize: (devChunkIndex+1)*devChunkSize] = (x_dev[devChunkIndex*devChunkSize: (devChunkIndex+1)*devChunkSize]-x_train_mean)/x_train_std
		else:
			x_dev[devChunkIndex*devChunkSize: ] = (x_dev[devChunkIndex*devChunkSize: ] - x_train_mean)/x_train_std
			

	## Need to do the same for test too
	return (x_train, y_train_oh), (x_dev, y_dev_oh), (x_test, y_test_oh)

class Classifier:
	def __init__(self, resultDir: str, modelName: str, x_train, y_train_oh, x_dev, y_dev_oh, x_test, y_test_oh, numPowerTraces):
		""" Initialize parameters and sequential model for training
		"""
		self.resultDir = resultDir
		self.modelName = modelName
		self.x_train = x_train[:,0:numPowerTraces]
		self.x_dev = x_dev[:, 0:numPowerTraces]
		self.x_test = x_test[:, 0:numPowerTraces]
		self.y_train_oh = y_train_oh
		self.y_dev_oh = y_dev_oh
		self.y_test_oh = y_test_oh
		self.numPowerTraces = numPowerTraces
		
	def createFNNModel(self, hiddenLayer, actList, dropList, batchNorm, numPowerTraces):
		self.hiddenLayer = hiddenLayer
		self.actList = actList
		self.dropList = dropList
		self.batchNorm = batchNorm
		self.numPowerTraces = numPowerTraces
		self.model = Sequential()

		## Building models
		for index, layer in enumerate(hiddenLayer):
			if (index == 0):
				## 1st layer
				self.model.add(Dense(hiddenLayer[index], activation=actList[index], input_shape=(numPowerTraces,)))
			else:
				self.model.add(Dense(hiddenLayer[index], activation=actList[index]))

			if (batchNorm[index]):
				self.model.add(BatchNormalization())
			
			self.model.add(Dropout(dropList[index]))

		## Last layer is softmax layer
		self.model.add(Dense(256, activation='softmax'))

		#self.model.compile(loss='categorical_crossentropy', metrics=['categorical_accuracy'], optimizer='adam')
		#print("Model summary\n")
		#print(self.model.summary())
		return self.model
	
	def createRNNModel(self, hiddenList, denseList, denseAct, rnnDropList, denseDropList, typeOfLayer, biDir, trainShape):
		self.model = Sequential()
		print("hiddenList= %s" %(hiddenList))
		#print("rnnDropList= %s\n recurrDropList= %s\n denseDropList= %s" %(rnnDropList, recurrDropList, denseDropList))
		print("rnnDropList= %s\ndenseDropList= %s" %(rnnDropList, denseDropList))
		for index, hiddenNum in enumerate(hiddenList):
			print("index=%s, hiddenNum=%s" %(index, hiddenNum))
			print("type\nhiddenNum=%s\nrnnDrop=%s\nbiDir=%s\n"\
			%(type(hiddenNum), type(rnnDropList[index]),type(biDir)))
			if(typeOfLayer == "GRU"):
				##Input Layer
				if(index==0):
					if(biDir):
						print("BiDir, %s, index=%s, hiddenNum=%s, trianShape=%s" %(typeOfLayer, index, hiddenNum, (1,self.numPowerTraces)))
						#self.model.add(Bidirectional(GRU(hiddenNum,dropout=rnnDropList[index], return_sequences=True), input_shape=trainShape[-2:]))
						self.model.add(Bidirectional(GRU(int(hiddenNum), return_sequences=True), input_shape=(1,self.numPowerTraces)))
					else:
						print("%s, index=%s, hiddenNum=%s, trianShape=%s" %(typeOfLayer, index, hiddenNum, (1,self.numPowerTraces)))
						#self.model.add(GRU(hiddenNum,dropout=rnnDropList[index],return_sequences=True, input_shape=trainShape[-2:]))
						self.model.add(GRU(int(hiddenNum),return_sequences=True, input_shape=(1,self.numPowerTraces)))
				else:
					if(biDir):
						print("BiDir, %s, index=%s, hiddenNum=%s" %(typeOfLayer, index, hiddenNum))
						#self.model.add(Bidirectional(GRU(hiddenNum,dropout=rnnDropList[index], return_sequences=True)))
						self.model.add(Bidirectional(GRU(int(hiddenNum), return_sequences=True)))
					else:
						print("%s, index=%s, hiddenNum=%s" %(typeOfLayer, index, hiddenNum))
						#self.model.add(GRU(hiddenNum,dropout=rnnDropList[index], return_sequences=True))
						self.model.add(GRU(int(hiddenNum), return_sequences=True))
			elif(typeOfLayer == "LSTM"):
				##Input Layer
				if(index==0):
					if(biDir):
						print("BiDir, %s, index=%s, hiddenNum=%s, trianShape=%s" %(typeOfLayer, index, hiddenNum, trainShape[-2:]))
						self.model.add(Bidirectional(LSTM(hiddenNum,dropout=rnnDropList[index], return_sequences=True), input_shape=trainShape[-2:]))
					else:
						print("%s, index=%s, hiddenNum=%s, trianShape=%s" %(typeOfLayer, index, hiddenNum, trainShape[-2:]))
						self.model.add(LSTM(hiddenNum,dropout=rnnDropList[index], return_sequences=True, input_shape=trainShape[-2:]))
				else:
					if(biDir):
						print("BiDir, %s, index=%s, hiddenNum=%s" %(typeOfLayer, index, hiddenNum))
						self.model.add(Bidirectional(LSTM(hiddenNum, dropout=rnnDropList[index], return_sequences=True)))
					else:
						print("%s, index=%s, hiddenNum=%s" %(typeOfLayer, index, hiddenNum))
						self.model.add(LSTM(hiddenNum,dropout=rnnDropList[index], return_sequences=True))
			else:
				print("typeOfLayer not GRU or LSTM\n")
		
		##Dense Layers
		print("densList=%s, denseAct=%s" %(denseList, denseAct))
		for denseIdx, denseNum in enumerate(denseList):
			print("Dense, num=%s, act=%s" %(denseNum, denseAct[denseIdx])) 
			self.model.add(Dense(denseNum,activation=denseAct[denseIdx])) 
			self.model.add(Dropout(denseDropList[denseIdx]))

		##Output Layer
		self.model.add(Dense(256, activation='softmax'))
		return self.model

	def createCNNModel(self, cnnFilters, kernelSizes, cnnActs, trainShape, batchNorm, isMaxPool, maxPoolSizes, denseList, denseActs, denseDropouts):
		self.model = Sequential()
		print("inside createCNNModel")
		print("cnnFilters=%s\nkernelSizes=%s\ncnnActs=%strainShape=%s\nbatchNorm=%s\nisMaxPool=%s\nmaxPoolSizes=%s\n" %(cnnFilters, kernelSizes, cnnActs, trainShape, batchNorm, isMaxPool, maxPoolSizes))
		for index, cnnFilter in enumerate(cnnFilters):
			if(index == 0):
				self.model.add(Conv1D(filters=cnnFilters[index], kernel_size=int(kernelSizes[index]), activation=cnnActs[index], input_shape=trainShape))
			else:
				self.model.add(Conv1D(filters=cnnFilters[index], kernel_size=int(kernelSizes[index]), activation=cnnActs[index]))
			##Add dropouts 
			if (batchNorm[index]):
				self.model.add(BatchNormalization())
			if(isMaxPool[index]):
				##Added avergae pooling for experiments
				#self.model.add(MaxPooling1D(pool_size=maxPoolSizes[index]))
				self.model.add(AveragePooling1D(pool_size=maxPoolSizes[index]))
		##Addd flatten layer
		self.model.add(Flatten())
		print("denseList=%s\ndenseActs=%s\ndenseDropouts=%s\n" %(denseList, denseActs, denseDropouts))
		for idx, denseLayer in enumerate(denseList):
			self.model.add(Dense(denseList[idx], activation=denseActs[idx]))
			self.model.add(Dropout(denseDropouts[idx]))

		##Output layer
		self.model.add(Dense(256, activation='softmax'))

		return self.model

	def createCNN2Model(self, cnnFilters, kernelSizes, cnnActs, trainShape, batchNorm, isMaxPool, maxPoolSizes, denseList, denseActs, denseDropouts):
		self.model = Sequential()
		print("inside createCNNModel")
		print("cnnFilters=%s\nkernelSizes=%s\ncnnActs=%strainShape=%s\nbatchNorm=%s\nisMaxPool=%s\nmaxPoolSizes=%s\n" %(cnnFilters, kernelSizes, cnnActs, trainShape, batchNorm, isMaxPool, maxPoolSizes))
		for index, cnnFilter in enumerate(cnnFilters):
			if(index == 0):
				self.model.add(Conv2D(filters=cnnFilters[index], kernel_size=int(kernelSizes[index]), activation=cnnActs[index], input_shape=trainShape))
			else:
				self.model.add(Conv2D(filters=cnnFilters[index], kernel_size=int(kernelSizes[index]), activation=cnnActs[index]))
			##Add dropouts 
			if (batchNorm[index]):
				self.model.add(BatchNormalization())
			if(isMaxPool[index]):
				##Added avergae pooling for experiments
				#self.model.add(MaxPooling1D(pool_size=maxPoolSizes[index]))
				self.model.add(AveragePooling1D(pool_size=maxPoolSizes[index]))
		##Addd flatten layer
		self.model.add(Flatten())
		print("denseList=%s\ndenseActs=%s\ndenseDropouts=%s\n" %(denseList, denseActs, denseDropouts))
		for idx, denseLayer in enumerate(denseList):
			self.model.add(Dense(denseList[idx], activation=denseActs[idx]))
			self.model.add(Dropout(denseDropouts[idx]))

		##Output layer
		self.model.add(Dense(256, activation='softmax'))

		return self.model

	def train(self, batchSize):
		""" Train the model with the training data
		batchSize : batch size during trainig
		"""

		Epochs = 1000

		logFile = self.resultDir + '/' + self.modelName + '_' + str(len(self.hiddenLayer)) + 'HL_'  + str(batchSize) +'.log'
		csv_logger = CSVLogger(logFile, append=True, separator="\t")
		
		earlyStop = EarlyStopping(monitor='val_categorical_accuracy', patience=10, mode='auto', verbose=1, restore_best_weights=True)
		
		##filePath = self.resultDir + '/' + self.modelName + '_checkPoint_best_model.hdf5'
		#### This file will include the epoch number when it gets saved.
		##repeatingFile = self.resultDir + '/' + self.modelName +'_{epoch:02d}_epoch_acc_{accVar:.2f}.hdf5'
		#### By default the every_10epochs will save the model at every 10 epochs
		##checkPoint = newCallBacks.ModelCheckpoint_every_10epochs(filePath, repeatingFile, self.x_test, self.y_test_oh , monitor='val_categorical_accuracy', verbose=1, save_best_only=True, every_10epochs=True)
		
		self.history = self.model.fit(self.x_train, self.y_train_oh, batch_size= batchSize, epochs=Epochs, verbose=1, shuffle= True, validation_data=(self.x_dev, self.y_dev_oh), callbacks=[csv_logger, earlyStop])

	def evaluate(self):
		""" Evaluate the model on itself
		"""

		## We should be evaluating on dev dataset as well, so commenting x_test
		#self.model_score = self.model.evaluate(self.x_test, self.y_test_oh, batch_size=2048)
		self.model_score = self.model.evaluate(self.x_dev, self.y_dev_oh, batch_size=2048)
		print("%s score = %f\n" %(self.modelName, self.model_score[1]))

		##Saving atucal vs predicted predictions
		##np.argmax returns the index where it see's 1 in the row
		#y_pred = np.argmax(self.model.predict(self.x_test, batch_size=2048), axis=1)
		y_pred = np.argmax(self.model.predict(self.x_dev, batch_size=2048), axis=1)

		## vstack will stack them in 2 rows, so we use Trasnpose to get them in column stack
		#output_predict = np.vstack((np.argmax(self.y_test_oh, axis=1), y_pred)).T
		output_predict = np.vstack((np.argmax(self.y_dev_oh, axis=1), y_pred)).T

		self.outputFile = self.resultDir + '/' + self.modelName + '_' + str(len(self.hiddenLayer)) + 'HL_'  + str(self.history.epoch[-1]+1) + 'epochs_' + '{0:.2f}'.format(self.model_score[1]*100).replace('.', 'p') + '_acc_' + "outputPredict.csv" 
		
		np.savetxt(self.outputFile, output_predict, fmt="%5.0f", delimiter=",")

		##Error Analysis of the prediction
		#errorAnalysis(self.outputFile)

		return self.model_score

	def saveModel(self):
		""" Save the model
		"""
		saveStr = self.resultDir + '/' + self.modelName + '_' + str(len(self.hiddenLayer)) + 'HL_'  + str(self.history.epoch[-1]+1) + 'epochs_' + '{0:.2f}'.format(self.model_score[1]*100).replace('.', 'p') + '_acc_' + '.h5'
		print("Saving model to\n%s\n" %(saveStr))
		self.model.save(saveStr)

	def keyAccuracy(self):

		df = pd.read_csv(self.outputFile, header=None)
		
		error_df = df[df[0]!=df[1]].astype('category')
		error_df[2] = error_df[0].astype(str).str.cat(error_df[1].astype(str), sep="-")
		
		totalCount = df[0].count()
		errorCount = error_df[2].count()
		accuracy = ((df[0].count()-error_df[2].count())/df[0].count())*100
		
		
		## to get the accuracy of individual keys, we need to count the number of rows in error_df for the same key
		## dubtract it from total data elements and divide it by the total number of data elements for each.
		
		## For example
		#c1 = df[0][df[0] == 0].count()
		#error_0 = error_df[0][error_df[0]==0].count()
		#
		#acc_0 = ((c1-error_0)/c1)*100
		
		## Now to loop it
		keyAcc = pd.DataFrame(columns={'key', 'acc'})
		
		for key in range(256):
			totalKey = df[0][df[0]==key].count()
			keyErrors = error_df[0][error_df[0]==key].count()
			acc = ((totalKey-keyErrors)/totalKey)*100
			keyAcc.loc[key] = {'key': key, 'acc': acc}
			#print("key= %s, acc= %s" %(key, acc))
		
		## Save to tsv
		saveFile = self.resultDir + '/' + self.modelName + '_' + str(len(self.hiddenLayer)) + 'HL_'  + str(self.history.epoch[-1]+1) + 'epochs_' + '{0:.2f}'.format(self.model_score[1]*100).replace('.', 'p') + '_acc_' + "keyAccuracy.tsv" 
		keyAcc.to_csv(saveFile, sep='\t', header=True, index=False)
		
		##Checking for entopy calculations of the predictions
		pred = self.model.predict(self.x_dev, batch_size=2048)
		
		## np.argsort(-pred) gets the order in which the indexes will be arranged 
		## np.argsort() of above will return the rank of the indexes with their values in the coressponding indexes
		rank = np.argsort(np.argsort(-pred))
		
		## Get the actual predictions from y_dev, need to convert one-hot to actual numbers
		dev_actual = np.argmax(self.y_dev_oh, axis=1)
		
		## Get the prediction ranks for each prediction
		prediction_ranks = rank[np.arange(len(dev_actual)), dev_actual]
		
		## getting the mean will also get the accuracy for the recall_at
		## recall = 1 , will get you accuracy at one shot
		recall_1 = np.mean(prediction_ranks < 1)
		recall_10 = np.mean(prediction_ranks < 10)
		recall_25 = np.mean(prediction_ranks < 25)
		recall_40 = np.mean(prediction_ranks < 40)
		recall_50 = np.mean(prediction_ranks < 50)
	
		print("model= %s\nrecall_1= %s\nrecall_10= %s\nrecall_25= %s\nrecall_40= %s\nrecall_50= %s" %(self.modelName, recall_1, recall_10, recall_25, recall_40, recall_50))

		## Create confusion matriux for aggregated computation
		conf = confusion_matrix(df[0], df[1])
		## Get the index for each row's max value. This is the column number in each row where the max value is located
		rowArgMax = np.argmax(conf, axis=1)

		logFile = self.resultDir + '/../log/' + self.modelName + '_' + str(len(self.hiddenLayer)) + 'HL_'  + str(self.history.epoch[-1]+1) + 'epochs_' + '{0:.2f}'.format(self.model_score[1]*100).replace('.', 'p') + '_acc_' + "keyAccuracy.log" 

		with open(logFile, 'a') as f:
			f.write("model= %s\nrecall_1= %s\nrecall_10= %s\nrecall_25= %s\nrecall_40= %s\nrecall_50= %s\n\n" %(self.modelName, recall_1, recall_10, recall_25, recall_40, recall_50))
			for row in range(256):
				if (row != rowArgMax[row]):
					print("key= %s (%s%% acc), maxPredKey= %s (%s%% acc)---" %(row, conf[row, row]/10, rowArgMax[row], conf[row, rowArgMax[row]]/10))
					f.write("key= %s (%s%% acc), maxPredKey= %s (%s%% acc)---\n" %(row, conf[row, row]/10, rowArgMax[row], conf[row, rowArgMax[row]]/10))
				else:
					print("key= %s (%s%% acc), maxPredKey= %s (%s%% acc)" %(row, conf[row, row]/10, rowArgMax[row], conf[row, rowArgMax[row]]/10))
					f.write("key= %s (%s%% acc), maxPredKey= %s (%s%% acc)\n" %(row, conf[row, row]/10, rowArgMax[row], conf[row, rowArgMax[row]]/10))

	def getPerplexity(self, x_dev, y_dev_oh, model):
		## Predict probabilities for each dev set
		pred = model.predict(x_dev, batch_size=2048)
		
		print("len of y_dev_oh = %s" %(len(y_dev_oh.shape)))
		if(len(y_dev_oh.shape) > 2):
			y_dev_oh = y_dev_oh.reshape(-1, 256)

		dev_actual = np.argmax(y_dev_oh, axis=1)
		dev_actual = dev_actual.reshape(dev_actual.shape[0], 1) #shape should be in array form for take_along_axis
		
		pred_prob = np.take_along_axis(pred, dev_actual, 1)
		
		## loop to calculate perplexity
		product = 1
		N = pred_prob.shape[0] #Total number of dev samples
		for index in range(N):
			## Take the n'th root of each value and then multiply them
			product = product * np.power(pred_prob[index], (1/N))
		
		perplexity = 1/product
		#Mprint("perplexity of model=%s on %s of %s is %s\n" %(self.saveStr.split("/")[-1], self.configName, dataset ,perplexity))
		print("perplexity is %s\n" %(perplexity))
		return perplexity



##https://stackoverflow.com/questions/56079223/custom-keras-data-generator-with-yield
class DataGenerator(tf.keras.utils.Sequence):
	def __init__(self, files, batch_size=32, shuffle=True, random_state=42, typeOfData="train", numPowerTraces=6000, typeOfModel="FNN", preprocess=False, preTrainModel=None):
		'Initialization'
		self.files = files ##list of files to load
		#self.labels = labels
		self.batch_size = batch_size ##Number of files in each batch
		self.shuffle = shuffle
		self.random_state = random_state
		self.typeOfData = typeOfData ##The key used to load data from files
		self.numPowerTraces = numPowerTraces ##This is used for CNN and RNN reshaping
		self.typeOfModel = typeOfModel ##Model tyep, FNN, CNN, RNN
		self.on_epoch_end()
		self.monitor_op=np.greater
		self.preprocess = preprocess
		self.preTrainModel=preTrainModel
		print("preprocess=%s" %(self.preprocess))

	def __len__(self):
		##This can be changed to totalFiles/batchSize
		# total_files = 256
		# batch_files = 32
		return int(np.floor(len(self.files)/self.batch_size))
		#return int(np.floor(len(self.files) / self.batch_size))

	def __getitem__(self, index):
		# Generate indexes of the batch
		indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
		files_batch = [self.files[k] for k in indexes]
		# Generate data
		if (self.typeOfData == "data"):
			x, y = self.__data_generation(files_batch)
		elif(self.typeOfData == "train"):
			x, y = self.__train_generation(files_batch)
		elif(self.typeOfData == "dev"):
			x, y = self.__dev_generation(files_batch)
		elif(self.typeOfData == "wavelet"):
			x, y = self.__wavelet_generator(files_batch)
			if(self.preTrainModel=="resnet50"):
				x = x.reshape(-1,127,1500,1)
				x = np.repeat(x,3,axis=-1)
		elif(self.typeOfData == "stft"):
			x, y = self.__stft_generator(files_batch)
			if(self.preTrainModel=="resnet50"):
				x = x.reshape(-1,129,48,1)
				x = np.repeat(x,3,axis=-1)
		return x, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.files))
		if self.shuffle == True:
			np.random.seed(self.random_state)
			np.random.shuffle(self.indexes)
		##Save model after each epoch
		#if self.monitor_op(current, self.best):
		#if self.verbose > 0:
		#io_utils.print_msg(
		#f"\nEpoch {epoch + 1}: {self.monitor} "
		#"improved "
		#f"from {self.best:.5f} to {current:.5f}, "
		#f"saving model to {filepath}"

	def __data_generation(self, files):
		count = 0 
		for dataFile in files:
			if(count == 0 ):
				x = np.load(dataFile)["data"][:,:-1]
				y = np.load(dataFile)["data"][:,-1]
			else:
				x = np.concatenate([x, np.load(dataFile)["data"][:,:-1]], axis=0)
				y = np.concatenate([y, np.load(dataFile)["data"][:,-1]], axis=0)
			count = count + 1

		##Sine we have -128 to 127 range, we will add 256 to convert it to 0-255
		add_index = (y<0)
		y[add_index] += 256

		##preprocess if applicable
		if(self.preprocess=="cw_l443_6000"):
			pklPath = "/xdisk/tosiron/manojgopale/xdisk/gem5DataCollection/cw_l443_6000_stdScaler_1875/preProcess.pkl"
			with open(pklPath, 'rb') as pklFile:
				pkl = pickle.load(pklFile)
			x = pkl["stdScaler_l443"].transform(x)
			print(f" Std Scaler Preprocessing {self.preprocess}")
		elif(self.preprocess=="cw_f303_6000"):
			pklPath = "/xdisk/tosiron/manojgopale/xdisk/gem5DataCollection/cw_f303_6000_stdScaler_1875/preProcess.pkl"
			with open(pklPath, 'rb') as pklFile:
				pkl = pickle.load(pklFile)
			x = pkl["stdScaler_f303"].transform(x)
			print(f" Std Scaler Preprocessing {self.preprocess}")
		elif(self.preprocess=="cw_l443_1500"):
			pklPath = "/xdisk/tosiron/manojgopale/xdisk/gem5DataCollection/cw_l443_1500_decimate/preProcess_7p5kadd.pkl"
			#pklPath = "/xdisk/tosiron/manojgopale/xdisk/gem5DataCollection/cw_l443_1500_decimate/preProcess.pkl"
			with open(pklPath, 'rb') as pklFile:
				pkl = pickle.load(pklFile)
			x = pkl["stdScaler_l443_7p5kadd"].transform(x)
			#x = pkl["stdScaler_l443_1500_decimate"].transform(x)
			print(f" Std Scaler Preprocessing {self.preprocess}")
		elif(self.preprocess=="cw_f415_1500"):
			pklPath = "/xdisk/tosiron/manojgopale/xdisk/gem5DataCollection/cw_f415_1500_decimate/preProcess.pkl"
			with open(pklPath, 'rb') as pklFile:
				pkl = pickle.load(pklFile)
			x = pkl["stdScaler_f415_1500_decimate"].transform(x)
			print(f" Std Scaler Preprocessing {self.preprocess}")
		#elif(self.preprocess=="cw_f303_l443_6000"):
		#	pklPath = "/xdisk/tosiron/manojgopale/xdisk/gem5DataCollection/cw_f303_l443_6000_1875/preProcess.pkl"
		#	with open(pklPath, 'rb') as pklFile:
		#		pkl = pickle.load(pklFile)
		#	x = pkl["stdScaler_f303_l443"].transform(x)

		##Standardize + reshape
		x, y = self.__reshape(self.typeOfModel, x, y)
		#sc = StandardScaler().fit(x_train)
		#x_train = sc.transform(x_train)
		#x_train = x_train.reshape(-1,numPowerTraces,1)
		#print("x_train shape= %s" %(x_train.shape,))
		#trainShape = x_train.shape[-2:]
		return x, y
	

	def __train_generation(self, files):
		count = 0 
		for dataFile in files:
			if(count == 0 ):
				x = np.load(dataFile)["train"][:,:-1]
				y = np.load(dataFile)["train"][:,-1]
			else:
				x = np.concatenate([x, np.load(dataFile)["train"][:,:-1]], axis=0)
				y = np.concatenate([y, np.load(dataFile)["train"][:,-1]], axis=0)
			count = count + 1

		##Sine we have -128 to 127 range, we will add 256 to convert it to 0-255
		add_index = (y<0)
		y[add_index] += 256

		##Standardize + reshape
		x, y = self.__reshape(self.typeOfModel, x, y)
		#sc = StandardScaler().fit(x_train)
		#x_train = sc.transform(x_train)
		#x_train = x_train.reshape(-1,numPowerTraces,1)
		#print("x_train shape= %s" %(x_train.shape,))
		#trainShape = x_train.shape[-2:]
		return x, y
	

	def __dev_generation(self, files):
		count = 0 
		for dataFile in files:
			if(count == 0 ):
				x = np.load(dataFile)["dev"][:,:-1]
				y = np.load(dataFile)["dev"][:,-1]
			else:
				x = np.concatenate([x, np.load(dataFile)["dev"][:,:-1]], axis=0)
				y = np.concatenate([y, np.load(dataFile)["dev"][:,-1]], axis=0)
			count = count + 1

		##Sine we have -128 to 127 range, we will add 256 to convert it to 0-255
		add_index = (y<0)
		y[add_index] += 256

		##Standardize + reshape
		x, y = self.__reshape(self.typeOfModel, x, y)
		#sc = StandardScaler().fit(x_train)
		#x_train = sc.transform(x_train)
		#x_train = x_train.reshape(-1,numPowerTraces,1)
		#print("x_train shape= %s" %(x_train.shape,))
		#trainShape = x_train.shape[-2:]
		return x, y
	
	def __wavelet_generator(self, files):
		count = 0 
		for dataFile in files:
			if(count == 0 ):
				x = np.load(dataFile)["data"][:,:-1]
				y = np.load(dataFile)["data"][:,-1].reshape(-1,1)
			else:
				x = np.concatenate([x, np.load(dataFile)["data"][:,:-1]], axis=0)
				y = np.concatenate([y, np.load(dataFile)["data"][:,-1].reshape(-1,1)], axis=0)
			count = count + 1

		x = maxabs_scale(x,axis=1)
		x = x.reshape(-1,500,500,1)[:,0:5,:,:]
		return x, y
	
	def __stft_generator(self, files):
		count = 0 
		for dataFile in files:
			if(count == 0 ):
				#x = np.load(dataFile)["data"][:,:-1].reshape(-1,129,48,1) #Reshaping after standardizing
				x = np.load(dataFile)["data"][:,:-1]
				y = np.load(dataFile)["data"][:,-1].reshape(-1,1)
			else:
				x = np.concatenate([x, np.load(dataFile)["data"][:,:-1]], axis=0)
				y = np.concatenate([y, np.load(dataFile)["data"][:,-1].reshape(-1,1)], axis=0)
			count = count + 1

		#x = x*200
		x = scale(x.astype("float64"),axis=1)
		x = x.reshape(-1,129,13,1)
		return x, y
	
	def __reshape(self, typeOfModel, x_data, y_data):
		##Commented if using standardized data
		#sc = StandardScaler().fit(x_data)
		#x_data = sc.transform(x_data)

		if(typeOfModel == "CNN"):
			x_data = x_data.reshape(-1,self.numPowerTraces,1)
		elif(typeOfModel == "RNN"):
			x_data = x_data.reshape(-1,1,self.numPowerTraces) ##batchSize, timesteps, features
			y_data = y_data.reshape(-1,1,1) ##When we use one-hot the shape should be y_data.reshape(-1,1,256)

		return x_data, y_data


####SCAAML model from 
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

# from scaaml.utils import display_config
# from scaaml.utils import get_num_gpu


class SCAAML_model():
	def block(self, x,
						filters,
						kernel_size=3,
						strides=1,
						conv_shortcut=False,
						activation="relu"):
		"""Residual block with pre-activation
		From: https://arxiv.org/pdf/1603.05027.pdf

		Args:
				x: input tensor.
				filters (int): filters of the bottleneck layer.
				kernel_size(int, optional): kernel size of the bottleneck layer.
				defaults to 3.
				strides (int, optional): stride of the first layer.
				defaults to 1.
				conv_shortcut (bool, optional): Use convolution shortcut if True,
				otherwise identity shortcut. Defaults to False.
				use_batchnorm (bool, optional): Use batchnormalization if True.
				Defaults to True.
				activation (str, optional): activation function. Defaults to "relu".

		Returns:
				Output tensor for the residual block.
		"""
		x = layers.BatchNormalization()(x)
		x = layers.Activation(activation)(x)
		if conv_shortcut:
			shortcut = layers.Conv1D(4 * filters, 1, strides=strides)(x)
		else:
			if strides > 1:
				shortcut = layers.MaxPooling1D(1, strides=strides)(x)
			else:
				shortcut = x
		x = layers.Conv1D(filters, 1, use_bias=False, padding="same")(x)
		x = layers.BatchNormalization()(x)
		x = layers.Activation(activation)(x)
		x = layers.Conv1D(filters,
											kernel_size,
											strides=strides,
											use_bias=False,
											padding="same")(x)
		x = layers.BatchNormalization()(x)
		x = layers.Activation(activation)(x)
		x = layers.Conv1D(4 * filters, 1)(x)
		x = layers.Add()([shortcut, x])
		return x

	def stack(self, x, filters, blocks, kernel_size=3, strides=2, activation="relu"):
		"""A set of stacked residual blocks.
		Args:
				filters (int): filters of the bottleneck layer.
				blocks (int): number of conv blocks to stack.
				kernel_size(int, optional): kernel size of the bottleneck layer.
				defaults to 3.
				strides (int, optional): stride used in the last block.
				defaults to 2.
				conv_shortcut (bool, optional): Use convolution shortcut if True,
				otherwise identity shortcut. Defaults to False.
				activation (str, optional): activation function. Defaults to "relu".

		Returns:
				tensor:Output tensor for the stacked blocks.
		"""
		x = self.block(x,
							filters,
							kernel_size=kernel_size,
							activation=activation,
							conv_shortcut=True)
		for _ in range(2, blocks):
			x = self.block(x, filters, kernel_size=kernel_size, activation=activation)
		x = self.block(x, filters, strides=strides, activation=activation)
		return x

	def Resnet1D(self, input_shape, mdl_cfg):  # pylint: disable=C0103
		pool_size = mdl_cfg["initial_pool_size"]
		filters = mdl_cfg["initial_filters"]
		block_kernel_size = mdl_cfg["block_kernel_size"]
		activation = mdl_cfg["activation"]
		dense_dropout = mdl_cfg["dense_dropout"]
		num_blocks = [
				mdl_cfg["blocks_stack1"], mdl_cfg["blocks_stack2"],
				mdl_cfg["blocks_stack3"], mdl_cfg["blocks_stack4"]
		]
		inputs = layers.Input(shape=input_shape)
		x = inputs
		# stem
		x = layers.MaxPool1D(pool_size=pool_size)(x)
		# trunk: stack of residual block
		for block_idx in range(4):
			filters *= 2
			x = self.stack(x,
								filters,
								num_blocks[block_idx],
								kernel_size=block_kernel_size,
								activation=activation)

		# head model: dense
		x = layers.GlobalAveragePooling1D()(x)
		for _ in range(1):
			x = layers.Dropout(dense_dropout)(x)
			x = layers.Dense(256)(x)
			x = layers.BatchNormalization()(x)
			x = layers.Activation(activation)(x)

		outputs = layers.Dense(256, activation="softmax")(x)
		model = Model(inputs=inputs, outputs=outputs)
		#model.summary()

#		lr = optim_cfg["lr"]
#		model.compile(loss=["categorical_crossentropy"],
#									metrics=["acc"],
#									optimizer=Adam(lr))
		return model

	def get_model(self, input_shape, config):
		"""Return an instantiated model based of the config provided.
		Args:
			config (dict): scald config.
		"""
		mdl_cfg = config["model_parameters"]
		#optim_cfg = config["optimizer_parameters"]
		# display_config("model", mdl_cfg)
		# display_config("optimizer", optim_cfg)
		return self.Resnet1D(input_shape, mdl_cfg)
		
#####------------------------------------#######


####------------Functions generating csv----------------####
## csvPAth: Path of the csv path that stores the predicted key vs actual key for cw run
## key: actual key, for which we want to find the breakup
## interval: number of dev traces we want to consider
def getMaxTracesperKey(csvPath, logFile, interval):
	#df= pd.read_csv("/xdisk/tosiron/manojgopale/extra/keyPrediction_chip/scr/moreDataTrials/result/rerun_177_4_031821_4HL_346epochs_58p77_acc_outputPredict.csv", header=None)
	df= pd.read_csv(csvPath ,header=None)
	print("csvFile= %s\n" %(csvPath))
	## Sort the first column in ascending order and also keep the order in which they appear in dev set
	## rename_axis names the index as "dfIndex"
	#df_sorted = df.rename_axis("index").sort_values(by=[0,"index"], ascending=["True", "True"])
	##For python3.5, use the below line
	df_sorted = df.rename_axis("index").sort_values(by=[0,"index"])
	#Mkey=12
	#Minterval=100
	count =0
	for key in range(256):
		print("key= %s\n" %(key))
		df_inter = df_sorted.loc[df_sorted.iloc[:,0] == key].iloc[0:interval,]
		df_inter.columns = ['actualKey','predictedKey']
		error_df = df_inter[df_inter["actualKey"]!=df_inter["predictedKey"]].astype('category')
		error_df["predictPairs"] = error_df["actualKey"].astype(str).str.cat(error_df["predictedKey"].astype(str), sep="-")
		totalCount = df_inter["actualKey"].count()
		errorCount = error_df["predictPairs"].count()
		accuracy = ((df_inter["actualKey"].count()-error_df["predictPairs"].count())/df_inter["actualKey"].count())*100
		accuracy
		## Get the number of predicted values for all the keys predicted.
		## Then get the max predicted key and compare it to the actual key. If they match then report the percentage and 
		## that it was predicted most number of times
		## as_index will create a different column instead of using the column '"predictedKey"' as the index
		df_group=df_inter.groupby(by="predictedKey", as_index=False).count()
		df_group.columns=["predictedKey", "count"]
		maxPredictedKey = df_group.loc[df_group["count"].values.argmax()]["predictedKey"]
		maxPrecitedKeyCount = df_group.loc[df_group["count"].values.argmax()]["count"]
		maxPredictedKeyPercentage = (maxPrecitedKeyCount/interval)*100

		##--- NOTE:Rank ---###
		##--Sort count columns inplace---###
		df_group.sort_values(by=["count"], ascending=False, ignore_index=True, inplace=True)
		##--Rank the keys based on its occurence from the top count---###
		df_group["rank"] = df_group["count"].rank(method="min", ascending=False)
		##-- Count and rank of the actual key ---###
#		pdb.set_trace()
		##-- Key might not be predicted at all in lower intervals --##
		if (key in df_group["predictedKey"].values):
			actualKeyCount = df_group.loc[df_group["predictedKey"]==key]["count"].values[0]
			actualKeyRank = df_group.loc[df_group["predictedKey"]==key]["rank"].values[0]
		else:
			actualKeyCount = 0
			actualKeyRank = 256 ##Not sure if 256 is correct, i need to think more on this
		##--Count and rank of the top predicted key --##
		topPredKeys = df_group.loc[df_group["rank"]==1]["predictedKey"].values.tolist() ##All top rank keys, if there are multiple keys
		topKeyCount = df_group.loc[df_group["rank"]==1]["count"].values[0]
		topKeyRank = df_group.loc[df_group["rank"]==1]["rank"].values[0]
		hammDist = distance.hamming([i for i in int2bin(key)], [i for i in int2bin(topPredKeys[0])])*8 ##-Hamming distance, multiplying by 8 to get number of different bits
		##--- End Rank ----##

		if (key == 0):
			with open(logFile, 'w') as f:
				f.write(f"key, predictedKey_{interval}, actualKeyCount_{interval}, actualKeyRank_{interval}, topKeyCount_{interval}, hammingDistance_{interval}\n")

		with open(logFile, 'a') as f:
			print(f"actualKeyCount= {actualKeyCount}, actualKeyRank= {actualKeyRank}, topPredKeys= {topPredKeys}, topKeyCount= {topKeyCount}, topKeyRank= {topKeyRank}, hammingDistance= {hammDist}")
			f.write(f"{key}, {' '.join(map(str,topPredKeys))}, {actualKeyCount}, {actualKeyRank}, {topKeyCount}, {hammDist}\n")
			if (key == maxPredictedKey):
				print("actual key is %s, maxPredicted key is %s\nmaxPredictedCount= %s, maxPredictedKeyPercentage= %s, interval= %s\n" %(key, maxPredictedKey, maxPrecitedKeyCount, maxPredictedKeyPercentage, interval))
#				f.write("actual key is %s, maxPredicted key is %s\nmaxPredictedCount= %s, maxPredictedKeyPercentage= %s, interval= %s\n" %(key, maxPredictedKey, maxPrecitedKeyCount, maxPredictedKeyPercentage, interval))
			else:
				count = count + 1
				print("###actual key is %s, maxPredicted key is %s\n###maxPredictedCount= %s, maxPredictedKeyPercentage= %s, interval= %s\n" %(key, maxPredictedKey, maxPrecitedKeyCount, maxPredictedKeyPercentage, interval))
#				f.write("###actual key is %s, maxPredicted key is %s\n###maxPredictedCount= %s, maxPredictedKeyPercentage= %s, interval= %s\n" %(key, maxPredictedKey, maxPrecitedKeyCount, maxPredictedKeyPercentage, interval))
	return count

####-----------Bitwise multi-trace analysis-------------------------####
def int2bin(x):
	return f'{x:08b}'

##df -> Dataframe with binary converted values
##output -> Dictionary with integer:probability for the dataframe
def getMaxBin(df):
	outputList = []
	columns = df.columns
	count = 0 ##Count how many times both 0's and 1's were equal
	ambgCol = [] ##store ambigous column numbers
	outputDict = {} ##Dictionary with integer: probability
	for col in columns:
		if (Counter(df[col])["0"] > Counter(df[col])["1"]):
			bitVal = 0
		##--NOTE: another parameter alpha -> fraction of 0's count to 1's count. Changing alpha we can see find an aplha that results in better binary outcomes
		##I see that the when the bits are 1's the 0 count is much closer to 1;s count. So we can play with this ratio to predict 1's when the absolute count of 0's is higher but close to 1's count. 
		elif (Counter(df[col])["0"] < Counter(df[col])["1"]):
			bitVal = 1
		else:
			count += 1
			bitVal = 1
			ambgCol.append(col)
		outputList.append(bitVal)
	#pdb.set_trace()
	##Create outputDict based on the number of possible outcomes
	if not ambgCol:
		outputDict[int("".join(str(i) for i in outputList),2)] = 1 ##Convert binary to integer
	else:
		##itertools.product creates tuples of all combinations of 0 and 1 with the length of ambgCol
		for tup in list(itertools.product([0, 1], repeat=len(ambgCol))):
			for num in range(len(ambgCol)):
				outputList[ambgCol[num]] = tup[num] ##Each tuple is a unique number that can be equal probability of prediction
			outputDict[int("".join(str(i) for i in outputList),2)] = 1/pow(2, len(ambgCol))
#M	print(f'Bit value equal for {count} digits, ambiguous columns = {ambgCol}, int= {int("".join(str(i) for i in outputList),2)}, outputList = {outputList}\noutputDict= {outputDict}')
	return outputDict


##df -> Dataframe with binary converted values
## interval -> number of traces taken nto find max binary output
## numIter -> numbeer oof iterations to perform
## finalDict -> Integer numbers with their respective probabilities
def intValCount(df, interval, numIter):
	finalDict = {} ##Stores dictionaries from getMaxBin and adds them with Counter
	for i in range(numIter):
		if(i == 0):
			finalDict = getMaxBin(df.sample(frac=1).iloc[:interval])
		else:
			finalDict = Counter(finalDict) + Counter(getMaxBin(df.sample(frac=1).iloc[:interval]))
	#print(f'{interval}: {finalDict}')
	##Convert values to ratios
	totalCount = sum(finalDict.values())
	for k, v in finalDict.items():
		finalDict[k] = v/totalCount*100
	return finalDict

##outputList -> Binray list for the integer
##ambgCol -> Ambiguous columns where 0 and 1 have the same probability
## Creates dictionary with all possible outcomes and probabilities for them
def allOutcomes(outputList, ambgCol):
	outputDict = {}
	##https://stackoverflow.com/questions/14931769/how-to-get-all-combination-of-n-binary-value
	print(list(itertools.product([0, 1], repeat=len(ambgCol))))
	for tup in itertools.product([0, 1], repeat=len(ambgCol)):
		for num in range(len(ambgCol)):
			print(f'tup= {tup}, num= {num}')
			outputList[ambgCol[num]] = tup[num] ##Each tuple is a unique number that can be equal probability of prediction
		print(f'outputList = {outputList}, integer= {int("".join(str(i) for i in outputList),2)}')
		outputDict[int("".join(str(i) for i in outputList),2)] = 1/pow(2, len(ambgCol))
	print(f'outputDict= {outputDict}')

##Integer Percentage  for the actual value.
def intPercentage(df, numIter, actualKey):
	percentageDict = {}
	##--Can have odd iterations for removing cases with equal values --##
	##-- np.linspace(1,601, 61) --##
	for i in np.linspace(1,601, 61):
		if(i==1):
			percentageDict[1] = intValCount(df, 1, numIter)[actualKey]/numIter
		else:
			percentageDict[int(i)] = intValCount(df, int(i), numIter)[actualKey]/numIter
	print(f'percentageDict = {percentageDict}')
	return percentageDict

####-----------------------------------------------------------####
#	##--To read
#	#loadDict = np.load(savePath, allow_pickle=True).item()

##model -> Model for which maxTrace analysis to perform
##x_dev, y_dev -> Development set to use
##saveDir -> If given, save the maxTrace.npy to the directory
def saveToDict(model, x_dev, y_dev, saveDir=None):
	pred = np.argmax(model.predict(x_dev, verbose=0).reshape(-1,256), axis=1)
	pred_analysis=np.concatenate([y_dev.reshape(len(y_dev),1),pred.reshape(len(pred),1)],axis=1)
	df = pd.DataFrame(data=pred_analysis)
	#intervalList=[5,10] ##Can be modified as per requirement
	intervalList = np.linspace(1,601, 61).astype(int)
	df_sorted = df.rename_axis("index").sort_values(by=[0,"index"])
	saveDict={} ##Dictionary to be filled
	numTrials = 10
	for interval in intervalList:
		saveDict[interval] = {}
		for key in range(256):
			saveDict[interval][key] = {}
			#print("key= %s\n" %(key))
			for num in range(numTrials):
				df_inter = df_sorted.loc[df_sorted.iloc[:,0] == key].sample(frac=1).iloc[0:interval,]
				df_inter.columns = ['actualKey','predictedKey']
				## Get the number of predicted values for all the keys predicted.
				## Then get the max predicted key and compare it to the actual key. If they match then report the percentage and 
				## that it was predicted most number of times
				## as_index will create a different column instead of using the column '"predictedKey"' as the index
				df_group=df_inter.groupby(by="predictedKey", as_index=False).count()
				df_group.columns=["predictedKey", "count"]
				##Create dictionary with {predKey: Count}
				perDict = {}
				for i, row in df_group.iterrows():
					perDict[row["predictedKey"]] = row["count"]/(numTrials*interval)
				if(num==0):
					finalDict=copy.deepcopy(perDict)
				else:
					finalDict=Counter(finalDict) + Counter(perDict)
			saveDict[interval][key] = finalDict
	#return saveDict
	##--Save file
	##--Save as JSON converts everything to string, difficult to use integers later.
	##--Save as npy, keeps all of them in same data type
	if(saveDir):
		savePath = saveDir + "/" +  "maxTrace.npy"
		np.save(savePath, saveDict)
		print(f"Saved file to {savePath}")
	return saveDict

##New function, 10.15.24. Not tested. Created saveToDict_parallel which will be used for runs
## Predict values based on maximum predictions in an interval for numTrails, then find the percentage accuracy of given key among the list of predictions
def saveToDict_new(model, x_dev, y_dev, saveDir=None):
	pred = np.argmax(model.predict(x_dev, verbose=0).reshape(-1,256), axis=1)
	intervalList = np.linspace(1,601, 61).astype(int)
	df_sorted = df.rename_axis("index").sort_values(by=[0,"index"])
	saveDict={} ##Dictionary to be filled
	numTrials = 10
	for interval in intervalList:
		saveDict[interval] = {}
		for key in range(256):
			predList = []
			saveDict[interval][key] = {}
			#print("key= %s\n" %(key))
			pred = np.argmax(model.predict(x_dev[y_dev==key], verbose=0).reshape(-1,256), axis=1)
			for num in range(numTrials):
				np.random.shuffle(pred) ##Shuffle prediction array
				##key_pred is the key that is predicted most number of time in the numTrials, key_count is the count of that key
				(key_pred, key_count) = Counter(np.argmax(pred[:interval,], axis=1)).most_common()[0]
				predList.append(key_pred)
			saveDict[interval][key] = Counter(predList)
	#return saveDict
	##--Save file
	##--Save as JSON converts everything to string, difficult to use integers later.
	##--Save as npy, keeps all of them in same data type
	if(saveDir):
		savePath = saveDir + "/" +  "maxTrace.npy"
		np.save(savePath, saveDict)
		print(f"Saved file to {savePath}")
	return saveDict

def maxTrace_interval(pred, interval, numTrials):
	predList = []
	for num in range(numTrials):
		np.random.shuffle(pred) ##Shuffle prediction array
		##key_pred is the key that is predicted most number of time in the numTrials, key_count is the count of that key
		(key_pred, key_count) = Counter(pred[:interval,]).most_common()[0]
		predList.append(key_pred)
	#print(f"predList= {predList}, interval={interval}")
	predCounter = Counter(predList)
	totalCount = sum(predCounter.values())
	for k, v in predCounter.items():
		predCounter[k] = v/totalCount*100
	return predCounter

def saveToDict_parallel(model, x_dev, y_dev, saveDir=None, num_tasks=20):
	#intervalList = np.linspace(1,601, 61).astype(int)
	intervalList = np.linspace(1,1401, 141).astype(int)
	numTrials = 10 
	print(f"num_tasks in the function = {num_tasks}")
	saveDict = {}
	for key in range(256): 
		saveDict[key] = {}
		#pred = np.argmax(model.predict(x_dev[y_dev==key], verbose=0))
		##reshape model.predict (-1,256) for RNN, works for FNN and CNN too
		pred = np.argmax(model.predict(x_dev[(y_dev==key).reshape(-1,)],batch_size=10240, verbose=0).reshape(-1,256), axis=1)
		#pred = pred.reshape(-1,1)
		with Pool(num_tasks) as pool:
			result = pool.starmap(maxTrace_interval, zip(repeat(pred), intervalList, repeat(numTrials)))
		##result will be output in the order of interval. Convert it to dictionary
		for inter, val in zip(intervalList, result):
			saveDict[key][inter] = val
		##Interchange inner(interval) and outer(key) keys of the dict
		interval = list(saveDict.values())[0].keys() ##Inner keys
		newDict = {inter: {key: saveDict[key][inter] for key in saveDict.keys()} for inter in interval}
	if(saveDir):
		savePath = saveDir + "/" +  "maxTrace.npy"
		np.save(savePath, newDict)
		print(f"Saved file to {savePath}")
	pprint.pprint(newDict, indent=4)
	return newDict

###---Save Avg Prob ---##
##model -> Model for which average probability analysis to perform
##x_dev, y_dev -> Development set to use
## Output --> {interval1: {0: {pred1: percentage1, pred2: percentage2}}, ...}
##saveDir -> If given, save the avgProb.npy to the directory
##NOTE: saveAvgProb_new -> Was modified on 10.15.24, but was not tested. Modified to include predictions and percentages, instead of softmax probabilities.
## saveAvgProb_parallel -> Created 10.15.24 has parallelizing functionality. Will be used for further runs
def saveAvgProb_new(model, x_dev, y_dev, saveDir=None):
	intervalList = np.linspace(1,601, 61).astype(int)
	saveDict = {}
	numTrials=10
	start=time.time()
	for interval in intervalList:
		saveDict[interval] = {}
		saveDict[interval][key] = {}
		startTime= time.time()
		for key in range(256):
			predList = [] ##Saves prediction for each key
			pred = model.predict(x_dev[(y_dev==key).reshape(-1,)],batch_size=10240, verbose=0)
			pred = pred.reshape(-1,256)
			##predList holds the predictions for each trial. I convert it to percentage while saving
			for num in range(numTrials):
				np.random.shuffle(pred)
				predList.append(np.argmax(np.average(pred[:interval], axis=0)))
			predCounter = Counter(predList) ##Save the count of each prediction
			##Convert count to percentage
			totalCount = sum(predCounter.values())
			for k, v in predCounter.items():
				predCounter[k] = v/totalCount*100
				
			saveDict[interval][key] = predCounter
#			print(f"Time for 10 trials= {time.time()-startTime} seconds")
#		print(f"interval= {interval}, Time for interval = {time.time()-startTime} seconds")
#	print(f"Total time taken = {time.time()-start}")
	if(saveDir):
		savePath = saveDir + "/" +  "avgProb.npy"
		np.save(savePath, saveDict)
		print(f"Saved file to {savePath}")
	return saveDict

##Returns a dictionary {pred1: percentage1, pred2: precentage2} for each key, interval pairs for numTrials
##percentage is just the ratio not multiplied by 100
def avgProb_interval(pred, interval, numTrials):
	predList = []
	##predList holds the predictions for each trial. I convert it to percentage while saving
	for num in range(numTrials):
		np.random.shuffle(pred)
		predList.append(np.argmax(np.average(pred[:interval], axis=0)))
	predCounter = Counter(predList) ##Save the count of each prediction
	##Convert count to percentage
	totalCount = sum(predCounter.values())
	for k, v in predCounter.items():
		predCounter[k] = v/totalCount*100
	return predCounter

def saveAvgProb_parallel(model, x_dev, y_dev, saveDir=None, num_tasks=20):
	#intervalList = np.linspace(1,601, 61).astype(int)
	intervalList = np.linspace(1,1401, 11).astype(int) #NOTE: comment this after testing
	#intervalList = np.linspace(1,1401, 141).astype(int)
	numTrials = 10 
	print(f"num_tasks in the function = {num_tasks}")
	saveDict = {}
	for key in range(256): #NOTE: Make key 256 after test
		startTime = time.time()
		saveDict[key] = {}
		#pred = np.argmax(model.predict(x_dev[y_dev==key], verbose=0).reshape(-1,256), axis=1)
		##Reshape (-1,256) for RNN, works for FNN and CNN
		pred = model.predict(x_dev[(y_dev==key).reshape(-1,)],batch_size=10240, verbose=0).reshape(-1,256)
		with Pool(num_tasks) as pool:
			result = pool.starmap(avgProb_interval, zip(repeat(pred), intervalList, repeat(numTrials)))
		##result will be output in the order of interval. Convert it to dictionary
		for inter, val in zip(intervalList, result):
			saveDict[key][inter] = val
		##Interchange inner(interval) and outer(key) keys of the dict
		interval = list(saveDict.values())[0].keys() ##Inner keys
		newDict = {inter: {key: saveDict[key][inter] for key in saveDict.keys()} for inter in interval}
		print(f"saveAvgProb_parallel: Done with key--{key} in time {time.time()-startTime} seconds")
	if(saveDir):
		savePath = saveDir + "/" +  "avgProb.npy"
		np.save(savePath, newDict)
		print(f"Saved file to {savePath}")
	pprint.pprint(newDict, indent=4)
	return newDict


def saveAvgProb(model, x_dev, y_dev, saveDir=None):
	intervalList = np.linspace(1,601, 61).astype(int)
	saveDict = {}
	numTrials=10
	start=time.time()
	for interval in intervalList:
#		saveDict[interval] = {}
		startTime= time.time()
		for key in range(256):
			pred = model.predict(x_dev[(y_dev==key).reshape(-1,)],batch_size=10240, verbose=0)
			pred = pred.reshape(-1,256)
			for num in range(numTrials):
				np.random.shuffle(pred)
				if(num==0):
					pred_avg=np.average(pred[:interval], axis=0).reshape(1,-1)
				else:
					pred_avg = np.concatenate((pred_avg, np.average(pred[:interval], axis=0).reshape(1,-1)), axis=0)
#			saveDict[interval][key] = np.average(pred_avg, axis=0).reshape(1,-1)
			##Concatenate all the average's as an array and then save them in the dictionary
			if(key==0):
				pred_save = np.average(pred_avg, axis=0).reshape(1,-1)
			else:
				pred_save = np.concatenate((pred_save, np.average(pred_avg, axis=0).reshape(1,-1)), axis=0)
#			print(f"Time for 10 trials= {time.time()-startTime} seconds")
		saveDict[interval] = pred_save
#		print(f"interval= {interval}, Time for interval = {time.time()-startTime} seconds")
#	print(f"Total time taken = {time.time()-start}")
	if(saveDir):
		savePath = saveDir + "/" +  "avgProb.npy"
		np.save(savePath, saveDict)
		print(f"Saved file to {savePath}")
	return saveDict

##--Save binary analysis

##Multiprocessing
## Multiprocessing with keeping interval constant and ranging keys did not work. model needs to be stored and shared across different 
## processes. Maybe taking too long.
## Will be trying keeping key constant and changing intervals
def binCalc(model, x_dev, y_dev, key, interval, numtrials):
	pred = np.argmax(model.predict(x_dev[y_dev==key], verbose=0).reshape(-1,256), axis=1)
	binary = pd.Series(pred).apply(int2bin)
	df_bin = pd.Series(binary).apply(list).apply(pd.Series)
	perDict = intValCount(df_bin, interval, numTrials)
	return perDict

def saveBinParallel(model, x_dev, y_dev, saveDir=None, num_tasks=20):
	intervalList = np.linspace(1,601, 61).astype(int)
	saveDict = {}
	numTrials = 10
	for interval in intervalList:
		saveDict[interval] = {}
		with Pool(num_tasks) as pool:
			result = pool.starmap(binCalc, zip(repeat(model), repeat(x_dev), repeat(y_dev), range(256), repeat(interval), repeat(numTrials)))
		##result has 256 different dictionaries coressponding to each key. Convert to dictionary
		for idx, val in enumerate(result):
			saveDict[interval][idx]=val
	if(saveDir):
		savePath = saveDir + "/" +  "binaryPerf.npy"
		np.save(savePath, saveDict)
		print(f"Saved file to {savePath}")
	pprint.pprint(saveDict, indent=4)
	return saveDict


##model -> Model for which binary analysis to perform
##x_dev, y_dev -> Development set to use
##saveDir -> If given, save the binaryPerf.npy to the directory
##num_tasks -> number of cpus to be used for parallel processing
def saveBinKey(model, x_dev, y_dev, saveDir=None, num_tasks=20):
	#intervalList = np.linspace(1,601, 61).astype(int)
	intervalList = np.linspace(1,1401, 141).astype(int)
	numTrials = 10
	print(f"num_tasks in the function = {num_tasks}")
	saveDict = {}
	for key in range(256):
		saveDict[key] = {}
		pred = np.argmax(model.predict(x_dev[(y_dev==key).reshape(-1,)], verbose=0).reshape(-1,256), axis=1)
		binary = pd.Series(pred).apply(int2bin)
		df_bin = pd.Series(binary).apply(list).apply(pd.Series)
		with Pool(num_tasks) as pool:
			result = pool.starmap(intValCount, zip(repeat(df_bin), intervalList, repeat(numTrials)))
		##result will be output in the order of interval. Convert it to dictionary
		for inter, val in zip(intervalList, result):
			saveDict[key][inter] = val
		##Interchange inner(interval) and outer(key) keys of the dict
		interval = list(saveDict.values())[0].keys() ##Inner keys
		newDict = {inter: {key: saveDict[key][inter] for key in saveDict.keys()} for inter in interval}
	if(saveDir):
		savePath = saveDir + "/" +  "binaryPerf.npy"
		np.save(savePath, newDict)
		print(f"Saved file to {savePath}")
	pprint.pprint(newDict, indent=4)
	return newDict
	
##model -> Model for which binary analysis to perform
##x_dev, y_dev -> Development set to use
##saveDir -> If given, save the binaryPerf.npy to the directory
def saveBin(model, x_dev, y_dev, saveDir=None):
	#intervalList=[5,301]
	intervalList = np.linspace(1,601, 61).astype(int)
	saveDict = {}
	numTrials = 10
	for interval in intervalList:
		saveDict[interval] = {}
		##To parallelize this loop, use multiprocessing Pool. 
		## Keep the interval constant, change the keys, output will be in the order of keys.
		## Will have to post process so that index of output is the key it relates to.
		for key in range(256):
			#saveDict[interval][key] = {}
			pred = np.argmax(model.predict(x_dev[y_dev==key], verbose=0).reshape(-1,256), axis=1)
			binary = pd.Series(pred).apply(int2bin)
			df_bin = pd.Series(binary).apply(list).apply(pd.Series)
			#perDict = intPercentage(df_bin, numTrials, key)
			perDict = intValCount(df_bin, interval, numTrials)
			#perDict = getMaxBin(df_bin.sample(frac=1).iloc[:interval])
			saveDict[interval][key] = perDict
	if(saveDir):
		savePath = saveDir + "/" +  "binaryPerf.npy"
		np.save(savePath, saveDict)
		print(f"Saved file to {savePath}")
	return saveDict

#####---functions for plotting trends----#######
##plotDict ->  {key1: {intv1: c1, intv2: c2..}, key2: {}} Dictionary of dictionary to be plot, 
##x axis is the interval for each entery and y axis is the value at that interval and the name is the key for which the entries was plot
##layoutDict -> update_layout parameter as a dict
##color -> Color to be used for the plot
##savePath -> Path where the plot is saved
def plotKeyTrend(plotDict, layoutDict, color, savePath):
	fig = go.Figure()
	for key in plotDict.keys():
		fig.add_trace(go.Scatter(x=list(plotDict[key].keys()), y=list(plotDict[key].values()), mode='lines', line=dict(color=color), name='key ' + str(key)))
	fig.update_layout(layoutDict)
	fig.write_html(savePath)

##Plot trend for a dictionary by interval
##plotDict -> {int1: c1, int2: c2, ..}
##layoutDict -> update_layout parameter as a dict
##color -> Color to be used for the plot
##savePath -> Path where the plot is saved
def plotTrend(plotDict, layoutDict, color, savePath):
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=list(plotDict.keys()), y=list(plotDict.values()), mode='lines', line=dict(color=color)))
	fig.update_layout(layoutDict)
	fig.write_html(savePath)

##Add trend to figures from different dictionaries
## fig-> Instance of plot that needs to be updates
## plotDict -> {int1, c1, int2: c2, ...} single dictionary
## color -> color of the plot
## name -> name of the plot
def plotAddTrend(fig, plotDict, color, name):
	fig.add_trace(go.Scatter(x=list(plotDict.keys()), y=list(plotDict.values()), mode='lines', line=dict(color=color), name=name))
	return fig

###----Accuracy 10.15.24 for modified scripts tha output {interval: {key1: {pred: percentage1, key2: percentage2}}
def getAccuracyMetric(maxDict):
	'''
	maxDict --> dictionary with {interval: {key1: {pred1: precntage1, pred2: precentage2..}}, interval2: {}}
	accDict --> dictionary {interval1: AvgAccuracy1, interval2: avgAccuracy2,...}
	keyAccDict --> dictionary {key1: [percentage1, percentage2...], key2: [..]} each value represents the performance for the interval going from 1 to 1401
	'''
	accDict = {}
	keyAccDict = {}
	for i in range(len(maxDict[1].keys())):
		keyAccDict[i] = [] ##Initialize each key as list, which will be appended with the interval accuracy
	##---Average accuracy of all keys for 10 trials ---##
	for interval in maxDict.keys():
		avg_acc = 0 #Average accuracy
		for key, val in enumerate(maxDict[interval].values()):
			if(key in val.keys()): ##Check if the key is in the prediction
				#print(f"key= {key}, val.keys= {val.keys()}")
				avg_acc += val[key]
				keyAccDict[key].append(val[key]) ##Add acccuracy for the particular key
			else:
				keyAccDict[key].append(0) ##Key is not predicted and accuracy for that key is 0
				#print(f"iavg_acc= {avg_acc}, percentage= {val[key]}")
		#print(f"avg_acc= {avg_acc/len(maxDict[interval].values())}, interval= {interval}")
		accDict[interval] = avg_acc/len(maxDict[interval].values())
	##---------------------------------------------------------##
	return accDict, keyAccDict


##--Accuracy
##maxDict -> Dictionary with interval: {key1: {key1: c1, ...}} output from saveToDict maxTraces
##output -> Dictionary {interval: Average Probability of key predicted}
def getAvgIntervalAcc(maxDict):
	accDict = {}
	for interval in maxDict.keys():
		correct = 0
		for k in maxDict[interval].keys():
			maxVal = max(maxDict[interval][k].values())
			##Get keys that are max values in the dict
			maxKeys = [key for key, val in maxDict[interval][k].items() if val == maxVal]
			if (k in maxKeys):
				#print(f"In maxKeys")
				if (len(maxKeys)==1):
					#print(f"In correct single")
					correct += 1 ##Add 1 to the total if the correct key is predicted
				else:
					#TODO: Is this even valid when checking the accuracy? should we say correct key was not predicted?
					#print(f"In correct multi")
					correct = correct + 1/len(maxKeys) ##Add 1/len(maxKeys) when multiple keys are predicted
		print(f"correct= {correct}, length= {len(maxDict[interval].keys())}")
		accDict[interval] = correct/len(maxDict[interval].keys())
	return accDict

##Average Rank does not yeild  much information
def getAvgIntervalRank(maxDict):
	rankDict = {}
	avgRankDict = {}
	rankKeyDict = {} ##Ranks of each key based on interval
	##Initialize dictionary with keys
	for i in range(256):
		rankKeyDict[i] = {}
	for interval in maxDict.keys():
		rankDict[interval] = {}
		sumRank = 0
		for k in maxDict[interval].keys():
			## Create a pandas Series, use rank method to rank the values, the create a dicionary {key1: rank1, key2: rank2..}
			rank = dict(zip(maxDict[interval][k].keys(), pd.Series(maxDict[interval][k].values()).rank(method='min', ascending=False).tolist()))
			rankDict[interval][k] = copy.deepcopy(rank)
			if(k in rank.keys()):
				sumRank += rank[k]
				rankKeyDict[k][interval] = rank[k]
			else:
				sumRank += 256
				rankKeyDict[k][interval] = 256 ##If key not in predicted values, rank is 256
		avgRankDict[interval] = sumRank/256
	return rankDict, avgRankDict, rankKeyDict

##----Avg Acc and Rank
def getAvgAccRank(maxDict):
	avgAccDict = {}
	rankDict = {}
	avgRankDict = {}
	rankKeyDict = {} ##Ranks of each key based on interval
	##Initialize dictionary with keys
	for i in range(256):
		rankKeyDict[i] = {}
	for interval in maxDict.keys():
		correct = 0
		rankDict[interval] = {}
		sumRank = 0
		for k in maxDict[interval].keys():
			##---Avg Acc
			maxVal = max(maxDict[interval][k].values())
			##Get keys that are max values in the dict
			maxKeys = [key for key, val in maxDict[interval][k].items() if val == maxVal]
			if (k in maxKeys):
				#print(f"In maxKeys")
				if (len(maxKeys)==1):
					#print(f"In correct single")
					correct += 1 ##Add 1 to the total if the correct key is predicted
				else:
					#TODO: Is this even valid when checking the accuracy? should we say correct key was not predicted?
					#print(f"In correct multi")
					correct = correct + 1/len(maxKeys) ##Add 1/len(maxKeys) when multiple keys are predicted
			##---Avg Rank
			## Create a pandas Series, use rank method to rank the values, the create a dicionary {key1: rank1, key2: rank2..}
			rank = dict(zip(maxDict[interval][k].keys(), pd.Series(maxDict[interval][k].values()).rank(method='min', ascending=False).tolist()))
			rankDict[interval][k] = copy.deepcopy(rank)
			if(k in rank.keys()):
				sumRank += rank[k]
				rankKeyDict[k][interval] = rank[k]
			else:
				sumRank += 256
				rankKeyDict[k][interval] = 256 ##If key not in predicted values, rank is 256
		print(f"interval= {interval}, correct= {correct}, length= {len(maxDict[interval].keys())}")
		avgAccDict[interval] = correct/len(maxDict[interval].keys())
		avgRankDict[interval] = sumRank/256
	return avgAccDict, rankDict, avgRankDict, rankKeyDict


##--Average probability plots --##
def getAvgProbParam(saveDict):
	##Average Accuracy
	avgAccDict = {}
	avgRankDict = {}
	rankDict = {}
	rankKeyDict = {}
	for interval in saveDict.keys():
		##If there dictionary is {intr1: [256x256], intr2: [256x256]}
		pred = np.argmax(saveDict[interval], axis=1)
		correct = 0
		for idx, key in enumerate(pred, start=0):
			if(idx==key):
				correct += 1
		avgAccDict[interval] = correct/256
		##Rank and Average rank
		avgRank = np.average(pd.DataFrame(data=saveDict[interval]).rank(method='min', ascending=False, axis=1).values.diagonal())
		avgRankDict[interval] = avgRank
		rankDict[interval] = dict(zip(range(256), pd.DataFrame(data=saveDict[interval]).rank(method='min', ascending=False, axis=1).values.diagonal().tolist()))
	##Key rank dict
	for i in range(256):
		rankKeyDict[i] = {}
	for interval in rankDict.keys():
		for k in rankDict[interval].keys():
			rankKeyDict[k][interval] = rankDict[interval][k]
	return avgAccDict, rankDict, avgRankDict, rankKeyDict

####----Accuracy per key-------####
## model -> model for which we want keywise accuracy
## x_Dev -> Development set
## y_dev -> Labels for development set, NOTE: Not one-hot encoded. one hot done within the functon
## return -> Dataframe with key and acc
def getAccPerKey(model, x_dev, y_dev):
	n_classes = 256
	y_dev_oh = np_utils.to_categorical(y_dev, n_classes)
	score_dev=model.evaluate(x_dev,y_dev_oh, batch_size=10240)
	print(f"score dev accuracy = {score_dev[1]}")
	pred = model.predict(x_dev, batch_size=10240)
	pred = pred.reshape(-1,256)
	pred_keys = np.argmax(pred, axis=1)
	
	pred_analysis=np.concatenate([y_dev.reshape(len(y_dev),1),pred_keys.reshape(len(pred_keys),1)],axis=1)
	
	df = pd.DataFrame(data=pred_analysis)
	
	error_df = df[df[0]!=df[1]].astype('category')
	error_df[2] = error_df[0].astype(str).str.cat(error_df[1].astype(str), sep="-")
	
	totalCount = df[0].count()
	errorCount = error_df[2].count()
	accuracy = ((df[0].count()-error_df[2].count())/df[0].count())*100
	
	keyAcc = pd.DataFrame(columns={'key', 'acc'})
	
	for key in range(256):
		totalKey = df[0][df[0]==key].count()
		keyErrors = error_df[0][error_df[0]==key].count()
		acc = ((totalKey-keyErrors)/totalKey)*100
		keyAcc.loc[key] = {'key': key, 'acc': acc}
		#print("key= %s, acc= %s" %(key, acc))
	return keyAcc

####---------------------------####
####------------------------------------------------------------------------------####	
#####------------------------------------#######
