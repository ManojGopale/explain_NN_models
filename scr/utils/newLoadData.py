import numpy as np
import pandas as pd
import gc, os, random, copy, json, collections, glob
import copy
from sklearn.utils import shuffle
from sklearn.preprocessing import RobustScaler, normalize, StandardScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
#from keras.utils import np_utils
## Python 3.9 and 3.10 version imports
import sys
if(sys.version_info > (3,9)):
	import keras.utils as np_utils
else:
	from keras.utils import np_utils

######-------Class for handling loads for devices------------#######
class Data:
	def processInput(self, pklPath, tracesPerKey):
		data = pd.read_pickle(pklPath)
		##Generate a randome integer to be used for shuffling
		random_int = np.random.randint(1000)
		print("random_int= %s, for shuffling data" %(random_int))
		##If an identical random integer is used during sample(frac=1, random_state), we should get the same sampling order
		## In case we dont, its still ok, since all the keys are for 1 particular key
		x_data = data.sample(frac=1, random_state=random_int).iloc[:tracesPerKey,:].trace.apply(pd.Series)
		y_data = data.sample(frac=1, random_state=random_int).iloc[:tracesPerKey,:].key.apply(lambda x: x[0])

		return x_data, y_data

	def getData(self, dataPath, tracesPerKey, dataSetType):
		##Load data for all keys
		x_data = pd.DataFrame()
		y_data = pd.DataFrame()

		for key in range(256):
			print("Started data processing for %d key\n" %(key))
			pklPath = dataPath + "/" + dataSetType + "_" + str(key) + ".pkl.zip"
			x_data_inter, y_data_inter = self.processInput(pklPath, tracesPerKey)
			x_data = pd.concat([x_data, x_data_inter], axis=0, ignore_index=True)
			y_data = pd.concat([y_data, y_data_inter], axis=0, ignore_index=True)

		return x_data, y_data
		
	def shuffleData(self, x_data, y_data):
		"""Shuffle the data, with random_state= None as default
		"""
		##NOTE: If there is any problem's with shuffling, go back to the utlis.shuffle function
		## Shuffling
		## https://scikit-learn.org/stable/modules/generated/sklearn.utils.shuffle.html
		#Mx_data, y_data = shuffle(x_data, y_data, random_state=None)

		##Commenting this for log readability
		#Mprint("\nStarted shuffling of data\nx_data[0]= %s\ny_data[0]= %s" %(x_data.iloc[0], y_data.iloc[0]))
		#Mprint(*x_data.iloc[0])
		#Mprint("\nx_data[100]= %s\ny_data[100]= %s" %(x_data.iloc[100], y_data.iloc[100]))
		#Mprint(*x_data.iloc[100])
		print("Before shuffling\ny_data[0]= %s\ny_data[100]= %s" %(y_data.iloc[0], y_data.iloc[100]))
		newIndex = np.random.permutation(x_data.index)
		## incase the index's from concat are duplicate, we will reset them before shuffling
		x_data.reset_index(drop=True, inplace=True)
		y_data.reset_index(drop=True, inplace=True)

		## the idea here is to shuffle the index keeping the order same and then reshuffling them to get the shuffled df
		## set_index sets the index's to the newIndex without changing the order.
		## we will have to sort them, so that the new order is randomized
		x_data.set_index(newIndex, inplace=True)
		y_data.set_index(newIndex, inplace=True)

		## Sorting indexes based on the index we just set
		x_data.sort_index(inplace=True)
		y_data.sort_index(inplace=True)

		#Mprint("\nFinished shuffling of data\nx_data[0]= %s\ny_data[0]= %s" %(x_data.iloc[0], y_data.iloc[0]))
		#Mprint(*x_data.iloc[0])
		#Mprint("\nx_data[100]= %s\ny_data[100]= %s" %(x_data.iloc[100], y_data.iloc[100]))
		#Mprint(*x_data.iloc[100])
		print("After shuffling\ny_data[0]= %s\ny_data[100]= %s" %(y_data.iloc[0], y_data.iloc[100]))
		return x_data, y_data
	
	def oneHotY(self, y_data):
		"""One hot the y_values in the dataset
		"""
		## One hot assignment
		n_classes = 256
		y_data_oh = np_utils.to_categorical(y_data, n_classes)
		
		print("\nOne-hot encoded for outputs\n")
		return y_data_oh
		
	def getStdParam(self, x_data):
		"""Get mean and std deviation of the x_data
		"""
		## get mean and std dev. for each column with axis=0
		x_data_mean = x_data.mean(axis=0)
		x_data_std  = x_data.std(axis=0)
		##Save initial mean and std for future use
		x_data_mean.to_csv("/xdisk/tosiron/manojgopale/extra/keyPrediction_chip/scr/moreDataTrials/result/summaryStats/mean_20000perkey.csv", index=False, header=False)
		x_data_std.to_csv("/xdisk/tosiron/manojgopale/extra/keyPrediction_chip/scr/moreDataTrials/result/summaryStats/std_20000perkey.csv", index=False, header=False)
		return x_data_mean, x_data_std
	
	def stdData(self, x_data, meanPath, stdPath):
		""" Standardize the x_data, by the mean and std deviation 
		Make sure the x_data is in numpy format
		meanPath, stdPath are the paths to the csv files storing the coressponding values
		"""
		##Mprint("Started normalizing\n")
		##M## since we are having mean and std dev. for each column, we will have to sample each row to apply them, hence axis=1
		##Mreturn x_data.apply(lambda x: (x-meanToApply)/stdToApply, axis=1).to_numpy()
		meanToApply = pd.read_csv(meanPath, header=None).to_numpy()
		stdToApply = pd.read_csv(stdPath, header=None).to_numpy()
		print("Loaded mean and std files from\n%s\n%s\n" %(meanPath, stdPath))
		## Reshaping so that it matches the standardization function and not error out
		meanToApply = meanToApply.reshape(meanToApply.shape[0], )
		stdToApply = stdToApply.reshape(stdToApply.shape[0], )

		print("Started normalizing\n")
		chunkSize = 28000
		if(x_data.shape[0]>chunkSize):
			chunkNum = int(x_data.shape[0]/chunkSize)
			for chunkIndex in range(chunkNum):
				print("chunkIndx= %s, chunkNum = %s" %(chunkIndex, chunkNum))
				if(chunkIndex != chunkNum-1): 
					x_data[chunkIndex*chunkSize: (chunkIndex+1)*chunkSize] = (x_data[chunkIndex*chunkSize: (chunkIndex+1)*chunkSize]-meanToApply)/stdToApply
				else:
					x_data[chunkIndex*chunkSize: ] = (x_data[chunkIndex*chunkSize: ] - meanToApply)/stdToApply
		else:
			print("ChunkSize less than 28000\n")
			x_data[0: ] = (x_data[0: ] - meanToApply)/stdToApply

		return x_data
		
	def fitTransform(self, x_data):
		"""
		Get the transform 
		MAke sure x_data is numpy.ndarray
		"""
		self.full_mean = x_data.mean()
		self.full_std = x_data.std(ddof=1)
		print("Full mean= %s, full std Dev = %s\n" %(self.full_mean, self.full_std))
		return (x_data-self.full_mean)/(self.full_std), self.full_mean, self.full_std
	
	def transform(self, x_data):
		"""
		Use the mean and std from fitTransform to standardize the data
		"""
		print("------transform ------\nFull mean= %s, full std Dev = %s\n" %(self.full_mean, self.full_std))
		return (x_data-self.full_mean)/(self.full_std)

######-----------------------------------------------------------#######



######-------Class for handling npz loads for devices------------#######
class manageData:
	def loadData(self, npzDir="/xdisk/tosiron/manojgopale/xdisk/gem5DataCollection/compressed_npyDatasets/", chipType="cw_f303_1500_15000", randInt=743, totalTraces="15k", byteNum=0):
		"""
		npzDir --->> Directory where npz files are stored
		chipType --->> Chip for which the data is to be loaded
		randInt --->> random integer for spitting the train/test
		totalTraces --->> number of traces used, 15k, 25k, 35k or 45k
		byteNum --->> return labels for specific byte of the key 0-15. Value of 16 will return all the key bytes in labels
		return --->> x_train, y_train, x_dev, y_dev
		"""
		if(chipType == "cw_f303_1500"):
			##To use complete 1500 trace data from CW.
		#	trainPath = npzDir + chipType + "_train.npz"
		#	print("Dataset picked from %s" %(trainPath))
		#	x_train = pd.DataFrame(data=np.load(trainPath)["x_train"])
		#	y_train = pd.DataFrame(data=np.load(trainPath)["y_train"])
		#	x_dev = pd.DataFrame(data=np.load(trainPath)["x_dev"])
		#	y_dev = pd.DataFrame(data=np.load(trainPath)["y_dev"])
		
			##For trials with decimate data
			#trainPath = npzDir + "/cw_f303_1500_1200tracesPerKey.npz"
			trainPath = npzDir + "/cw_f303_1500_15kAdd.npz"
			#trainPath = npzDir + "/cw_f303_1500_decimate_1200tracesPerKey.npz"
			data = np.load(trainPath)["train"]
			print(f"Data picked from {trainPath}")
			train_data, dev_data = train_test_split(data, test_size=.15, random_state=randInt)
			x_train = pd.DataFrame(data=train_data[:,:-1])
			y_train = pd.DataFrame(data=train_data[:,-1])
			x_dev =   pd.DataFrame(data=dev_data[:,:-1])
			y_dev =   pd.DataFrame(data=dev_data[:,-1])
		elif(chipType=="cw_f303_6000"):
			trainPath = npzDir + chipType + "_convert_train.npz"
			print("Dataset picked from %s" %(trainPath))
		
			##Load data
			trainData = pd.DataFrame(data=np.load(trainPath)["train"])
			trainData = trainData.sample(frac=1)
			x_train = trainData.iloc[:,:-1]
			y_train = trainData.iloc[:, -1]
		
			devData = pd.DataFrame(data=np.load(trainPath)["dev"])
			x_dev = devData.iloc[:,:-1]
			y_dev = devData.iloc[:, -1]
			#Mx_train = pd.DataFrame(data=np.load(trainPath)["train"][:,:-1])
			#My_train = pd.DataFrame(data=np.load(trainPath)["train"][:,-1])
			#Mx_dev = pd.DataFrame(data=np.load(trainPath)["dev"][:,:-1])
			#My_dev = pd.DataFrame(data=np.load(trainPath)["dev"][:,-1])
		elif(chipType == "cw_l443_1500"):
			trainPath = npzDir + "/cw_l443_1500_decimate_1200tracesPerKey.npz"
			data = np.load(trainPath)["train"]
			print(f"Data picked from {trainPath}")
			train_data, dev_data = train_test_split(data, test_size=.15)
			x_train = pd.DataFrame(data=train_data[:,:-1])
			y_train = pd.DataFrame(data=train_data[:,-1])
			x_dev =   pd.DataFrame(data=dev_data[:,:-1])
			y_dev =   pd.DataFrame(data=dev_data[:,-1])
		elif(chipType == "cw_l443_1500_decimate_wavelet"):
			trainPath = npzDir + "/cw_l443_decimate_wavelet_2ndOrder_1200tracesPerKey.npz"
			data = np.load(trainPath)["train"]
			print(f"Data picked from {trainPath}")
			train_data, dev_data = train_test_split(data, test_size=.15)
			x_train = pd.DataFrame(data=train_data[:,:-1])
			y_train = pd.DataFrame(data=train_data[:,-1])
			x_dev =   pd.DataFrame(data=dev_data[:,:-1])
			y_dev =   pd.DataFrame(data=dev_data[:,-1])
		elif(chipType == "cw_f415_1500"):
			#trainPath = npzDir + "/cw_f415_1500_decimate_1200tracesPerKey.npz"
			#trainPath = npzDir + "/cw_f415_1500_012524.npz" ##Full data size is 15000 per key.
			#trainPath = npzDir + "/cw_f415_1500_ir_disabled_030424.npz" ##Full data size is 15000 per key.
			trainPath = npzDir + "/cw_f415_1500_ir_disabled_allCap_031224.npz"
			data = np.load(trainPath)["train"]
			##-------------Amplitude limiting the data-----------------##
			data[:,:-1] = np.where(data[:,:-1]> 0.26,  0.26, data[:,:-1])
			data[:,:-1] = np.where(data[:,:-1]<-0.03, -0.03, data[:,:-1])
			##--------------------------------------------------------##
			print(f"Data picked from {trainPath}")
			train_data, dev_data = train_test_split(data, test_size=.10, random_state= randInt)
			x_train = pd.DataFrame(data=train_data[:,:-1])
			y_train = pd.DataFrame(data=train_data[:,-1])
			x_dev =   pd.DataFrame(data=dev_data[:,:-1])
			y_dev =   pd.DataFrame(data=dev_data[:,-1])
		elif(chipType == "cw_f215_1500"):
			trainPath = npzDir + "/cw_f215_1500_ir_disabled_031824.npz"
			data = np.load(trainPath)["train"]
			###-------------Amplitude limiting the data-----------------##
			#data[:,:-1] = np.where(data[:,:-1]> 0.26,  0.26, data[:,:-1])
			#data[:,:-1] = np.where(data[:,:-1]<-0.03, -0.03, data[:,:-1])
			###--------------------------------------------------------##
			print(f"Data picked from {trainPath}")
			train_data, dev_data = train_test_split(data, test_size=.10, random_state= randInt)
			x_train = pd.DataFrame(data=train_data[:,:-1])
			y_train = pd.DataFrame(data=train_data[:,-1])
			x_dev =   pd.DataFrame(data=dev_data[:,:-1])
			y_dev =   pd.DataFrame(data=dev_data[:,-1])
		elif(chipType == "cw_f051_1500"):
			##Added for all key byte data on 11.21.24
			trainPath = npzDir + "/cw_f051_1500_032224_15k_allKeyBytes_0.npz"
			data = np.load(trainPath)["data"]

			##Journal paper completed with cw_f051_1500_032224.npz
			#trainPath = npzDir + "/cw_f051_1500_032224.npz"
			#data = np.load(trainPath)["train"]
			if("15k" not in totalTraces):
				trainPath = npzDir + "/cw_f051_2100_10kAdd_041624.npz"
				data = np.concatenate((data,np.load(trainPath)["train"]), axis=0)
			if("35k" in totalTraces or "45k" in totalTraces):
				trainPath = npzDir + "/cw_f051_2100_10kAdd_043024.npz"
				data = np.concatenate((data,np.load(trainPath)["train"]), axis=0)
			if("45k" in totalTraces):
				trainPath = npzDir + "/cw_f051_2100_10kAdd_061024.npz"
				data = np.concatenate((data,np.load(trainPath)["train"]), axis=0)
		
			###-------------Amplitude limiting the data-----------------##
			#data[:,:-1] = np.where(data[:,:-1]> 0.26,  0.26, data[:,:-1])
			#data[:,:-1] = np.where(data[:,:-1]<-0.03, -0.03, data[:,:-1])
			###--------------------------------------------------------##
			print(f"Data picked from {trainPath}\ndatasize={data.shape}")
			train_data, dev_data = train_test_split(data, test_size=.10, random_state= randInt)
			x_train = pd.DataFrame(data=train_data[:,:2100])
			#y_train = pd.DataFrame(data=train_data[:,-1])
			x_dev =   pd.DataFrame(data=dev_data[:,:2100])
			#y_dev =   pd.DataFrame(data=dev_data[:,-1])
			##Added on 11.21.24 to get specific key labels
			if(byteNum <= 16 and byteNum >=0 ):
				if(byteNum == 16):
					#Return all bytes of the key
					y_train = pd.DataFrame(data=train_data[:,2100:])
					y_dev =   pd.DataFrame(data=dev_data[:,2100:])
				else:
					#Return byte corresponding to byteNum value
					y_train = pd.DataFrame(data=train_data[:,2100+byteNum])
					y_dev =   pd.DataFrame(data=dev_data[:,2100+byteNum])
			else:
				print(f"byteNum value should be between 0 and 16 (inclusive). Entered byteNum value is {byteNum}")
				raise NameError(f"Enter correct byteNum value (0<= byteNum <= 16). Entered value {byteNum}")
		elif(chipType == "cw_f303_l443_1500"):
			trainPath_l443 = npzDir + "/cw_l443_1500_decimate_1200tracesPerKey.npz"
			data_l443 = np.load(trainPath_l443)["train"]
			trainPath_f303 = npzDir + "/cw_f303_1500_1200tracesPerKey.npz"
			#trainPath_f303 = npzDir + "/cw_f303_1500_decimate_1200tracesPerKey.npz"
			data_f303 = np.load(trainPath_f303)["train"]
			print(f"Data picked from {trainPath_f303} and {trainPath_l443}")
			##Split and add data from 2 devices with a known ratio
			f303_ratio = f303_ratio
			l443_ratio = 1 - f303_ratio
			print(f"f303 ratio= {f303_ratio}, l443_ratio= {l443_ratio}")
			train_data_f303, dev_data_f303 = train_test_split(data_f303, test_size=.15)
			train_data_l443, dev_data_l443 = train_test_split(data_l443, test_size=.15)
			f303_idx = np.random.choice(int(len(train_data_f303)) , int(np.floor(f303_ratio*len(train_data_f303))), replace=False)
			l443_idx = np.random.choice(int(len(train_data_l443)) , int(np.floor(l443_ratio*len(train_data_l443))), replace=False)
			train_data = np.concatenate([train_data_f303[f303_idx], train_data_l443[l443_idx]], axis=0)
			dev_data = np.concatenate([dev_data_f303, dev_data_l443], axis=0)
			rng = np.random.default_rng()
			rng.shuffle(train_data) ##shuffle data before training
			x_train = pd.DataFrame(data=train_data[:,:-1])
			y_train = pd.DataFrame(data=train_data[:,-1])
			x_dev =   pd.DataFrame(data=dev_data[:,:-1])
			y_dev =   pd.DataFrame(data=dev_data[:,-1])
		elif(chipType == "cw_f303_tinyAES_7000"):
			trainPath = npzDir + "/cw_f303_tineyAES_7000_train.npz"
			data = np.load(trainPath)["train"]
			collection = collections.Counter(data[:,-1])
			print(collection)
			print(f"Data picked from {trainPath}")
			train_data, dev_data = train_test_split(data, test_size=.15)
			x_train = pd.DataFrame(data=train_data[:,:numPowerTraces])
			y_train = pd.DataFrame(data=train_data[:,-1])
			x_dev =   pd.DataFrame(data=dev_data[:,:numPowerTraces])
			y_dev =   pd.DataFrame(data=dev_data[:,-1])
		elif(chipType == "cw_l443_tinyAES_7000"):
			trainPath = npzDir + "/cw_l443_tineyAES_7000_train.npz"
			data = np.load(trainPath)["train"]
			collection = collections.Counter(data[:,-1])
			print(collection)
			print(f"Data picked from {trainPath}")
			train_data, dev_data = train_test_split(data, test_size=.15)
			x_train = pd.DataFrame(data=train_data[:,:-1])
			y_train = pd.DataFrame(data=train_data[:,-1])
			x_dev =   pd.DataFrame(data=dev_data[:,:-1])
			y_dev =   pd.DataFrame(data=dev_data[:,-1])
		elif(chipType=="cw_f303_1500_15000"):
		#M	######-----------Load Data from pkl---------------######
		#M	dataDir =     "/xdisk/tosiron/manojgopale/extra/chipWhisperer_data/trace_key_1500/"
		#M	#moreDataDir = "/xdisk/tosiron/manojgopale/extra/chipWhisperer_data/trace_key_1500_1/"
		#M	data=newLoadData.Data()
		#M	x_train, y_train = data.getData(dataDir, trainSize, "train")
		#M	x_train, y_train = data.shuffleData(x_train, y_train)
		#M	
		#M	x_dev, y_dev = data.getData(dataDir, devSize, "dev")
		#M	x_dev, y_dev = data.shuffleData(x_dev, y_dev)
		#M	######-----------Load Data from pkl---------------######
			######-----------Load Data from compressed Dataset---------------######
			#trainPath = npzDir + "/cw_f303_1500_15kAdd.npz"
			#data = np.load(trainPath)["train"]
			#Added on 11.21.24 for entire key byte 
			#trainPath = npzDir + "/cw_f303_1500_15kAdd_082624_allKeyBytes_0.npz"
	
			#Key byte '0' training, Journal paper done with cw_f303_1500_15kAdd_082624_0.npz data
			trainPath = npzDir + "/cw_f303_1500_15kAdd_082624_0.npz"
			data = np.load(trainPath)["data"]

			##081324 is collected from ufo board, has lot of noise and not trainable. Use 082024 data instead, collected from cw-lite
			#trainPath = npzDir + "/cw_f303_1500_15kAdd_081324_0.npz"
			#data = np.load(trainPath)["data"]
			collection = collections.Counter(data[:,-1])
			if("15k" not in totalTraces):
				trainPath = npzDir + "/cw_f303_1500_10kAdd_082024_0.npz"
				data = np.concatenate((data,np.load(trainPath)["data"]), axis=0)
			if("35k" in totalTraces or "45k" in totalTraces):
				trainPath = npzDir + "/cw_f303_1500_10kAdd_082024_1.npz"
				data = np.concatenate((data,np.load(trainPath)["data"]), axis=0)
			if("45k" in totalTraces):
				trainPath = npzDir + "/cw_f303_1500_10kAdd_082024_2.npz"
				data = np.concatenate((data,np.load(trainPath)["data"]), axis=0)
			#MtrainPath = npzDir + "/cw_f303_1500_10kAdd_080824_0.npz"
			#Mdata = np.load(trainPath)["data"]
			#Mif("20k" in totalTraces or "30k" in totalTraces):
			#M	trainPath = npzDir + "/cw_f303_1500_10kAdd_080824_1.npz"
			#M	data = np.concatenate((data,np.load(trainPath)["data"]), axis=0)
			#Mif("30k" in totalTraces):
			#M	trainPath = npzDir + "/cw_f303_1500_10kAdd_080824_2.npz"
			#M	data = np.concatenate((data,np.load(trainPath)["data"]), axis=0)
		
			collection = collections.Counter(data[:,-1])
			print(collection)
			print(f"Data picked from {trainPath}")
			#train_data, dev_data = train_test_split(data, test_size=.10, random_state= 743) ##Added random_state on 9/14/23 for reproducible results
			train_data, dev_data = train_test_split(data, test_size=.10, random_state= randInt) ##Added random_state on 9/14/23 for reproducible results
			x_train = pd.DataFrame(data=train_data[:,:1500])
			#y_train = pd.DataFrame(data=train_data[:,-1])
			x_dev =   pd.DataFrame(data=dev_data[:,:1500])
			#y_dev =   pd.DataFrame(data=dev_data[:,-1])
			##Added on 11.21.24 to get specific key labels
			if(byteNum <= 16 and byteNum >=0 ):
				if(byteNum == 16):
					#Return all bytes of the key
					y_train = pd.DataFrame(data=train_data[:,1500:])
					y_dev =   pd.DataFrame(data=dev_data[:,1500:])
				else:
					#Return byte corresponding to byteNum value
					y_train = pd.DataFrame(data=train_data[:,1500+byteNum])
					y_dev =   pd.DataFrame(data=dev_data[:,1500+byteNum])
			else:
				print(f"byteNum value should be between 0 and 16 (inclusive). Entered byteNum value is {byteNum}")
				raise NameError(f"Enter correct byteNum value (0<= byteNum <= 16). Entered value {byteNum}")
			######-----------Load Data from compressed Dataset---------------######
		elif(chipType == "cw_l443_1500_15000"):
			#trainPath = npzDir + "/cw_l443_1500_15kAdd.npz"
			#data = np.load(trainPath)["train"]
			##All key byte data added on 11.21.24
			trainPath = npzDir + "/cw_l443_1500_15kAdd_081424_allKeyBytes_0.npz"

			##Journal paper done with cw_l443_1500_15kAdd_081424_0.npz
			#trainPath = npzDir + "/cw_l443_1500_15kAdd_081424_0.npz"
			data = np.load(trainPath)["data"]
			if("15k" not in totalTraces):
				trainPath = npzDir + "/cw_l443_1500_10kAdd_081124_0.npz"
				data = np.concatenate((data,np.load(trainPath)["data"]), axis=0)
			if("35k" in totalTraces or "45k" in totalTraces):
				trainPath = npzDir + "/cw_l443_1500_10kAdd_081124_1.npz"
				data = np.concatenate((data,np.load(trainPath)["data"]), axis=0)
			if("45k" in totalTraces):
				trainPath = npzDir + "/cw_l443_1500_10kAdd_081124_2.npz"
				data = np.concatenate((data,np.load(trainPath)["data"]), axis=0)
		
			#M##----Newly collected data, take 15k traces/key-----##
			#MtrainPath = npzDir + "/cw_l443_1500_10kAdd_081124_0.npz"
			#Mdata = np.load(trainPath)["data"]
			#MtrainPath = npzDir + "/cw_l443_1500_10kAdd_081124_1.npz"
			#Mdata = np.concatenate((data,np.load(trainPath)["data"]), axis=0)
			#Mdata = data[:int(15000*256),:]
			#M##--------------------------------------------------##
			collection = collections.Counter(data[:,-1])
			print(collection)
			print(f"Data picked from {trainPath}")
			train_data, dev_data = train_test_split(data, test_size=.10, random_state= 743)
			x_train = pd.DataFrame(data=train_data[:,:1500])
			#y_train = pd.DataFrame(data=train_data[:,-1])
			x_dev =   pd.DataFrame(data=dev_data[:,:1500])
			#y_dev =   pd.DataFrame(data=dev_data[:,-1])
			##Added on 11.21.24 to get specific key labels
			if(byteNum <= 16 and byteNum >=0 ):
				if(byteNum == 16):
					#Return all bytes of the key
					y_train = pd.DataFrame(data=train_data[:,1500:])
					y_dev =   pd.DataFrame(data=dev_data[:,1500:])
				else:
					#Return byte corresponding to byteNum value
					y_train = pd.DataFrame(data=train_data[:,1500+byteNum])
					y_dev =   pd.DataFrame(data=dev_data[:,1500+byteNum])
			else:
				print(f"byteNum value should be between 0 and 16 (inclusive). Entered byteNum value is {byteNum}")
				raise NameError(f"Enter correct byteNum value (0<= byteNum <= 16). Entered value {byteNum}")
		elif(chipType == "cw_f415_1500_fft"):
			savef415 = npzDir + "cw_f415_1500_012524_rfft.npz"
		#M	#####---------Load data from all files-----------------######
		#M	#trainList = np.random.choice([x for x in glob.glob(dataDir + 'cw_l443_6000_1875/train_*.npz')], 10, replace=False )
		#M	for idx, path in enumerate([x for x in glob.glob('/xdisk/tosiron/manojgopale/xdisk/gem5DataCollection/cw_f415_1500_012524_rfft/train_*.npz')]):
		#M		if (idx==0):
		#M			data = np.load(path)["data"]
		#M		else:
		#M			inter = np.load(path)["data"]
		#M			data = np.vstack((data, inter)) ##was hstack, data shape 238000,512, corrected on 2/7/24 to vstack
		#M	
		#M	###----Save to npz after combining----###
		#M	if(not os.path.isfile(savef415)):
		#M		np.savez_compressed(savef415, train= data)
		#M		print(f"f415 fft save at {savef415}")
		#M	###-----------------------------------###
		
			data = np.load(savef415)["train"]
			##DC component is at index=0, not taken for 1D CNN
			data = data[:,1:]
			print(f"data shape after dc component not considered = {data.shape}")
			print(f"Last 10 keys \n{data[-10:,-1]}")
			train_data, dev_data = train_test_split(data, test_size=.10, random_state= randInt)#, random_state= 743)
			x_train = pd.DataFrame(data=train_data[:,:-1])
			y_train = pd.DataFrame(data=train_data[:,-1])
			x_dev =   pd.DataFrame(data=dev_data[:,:-1])
			y_dev =   pd.DataFrame(data=dev_data[:,-1])
		
		else:
			print(f"Current chipType={chipType}\nProvide correct chipType.")
			raise NameError(f"chipType not defined. Got invalid {chipType}\nx_train, y_train, x_dev, y_dev did not initialize")

		##Only check for negative labels when single byte is specified as label
		if(byteNum<=15 and byteNum>=0):
			##If labels are negative
			train_change=(y_train<0)
			dev_change = (y_dev<0)
			
			y_train[train_change] +=256
			y_dev[dev_change] += 256

		return x_train, y_train, x_dev, y_dev

	def processData(self, x_train, x_dev, typeOfStd="stdScaler"):
		"""
		Pro-process data based on the required processing
		"""
		if("stdScaler" in typeOfStd):
			print("Start standard scaling\n")
			##For more training data
			#x_train = np.vstack((x_train, x_dev))
			scaler = StandardScaler()
			x_train = scaler.fit_transform(x_train)
			x_dev   = scaler.transform(x_dev)
			if("stdScaler_5" in typeOfStd):
				print("Starting to cutoff standardized signal between -5 and +5")
				x_train = np.where(x_train>5, 5, x_train)
				x_train = np.where(x_train<-5,-5, x_train)
				x_dev = np.where(x_dev>5, 5, x_dev)
				x_dev = np.where(x_dev<-5, -5, x_dev)
				print("Completed cutoff standardization between -5 and +5")
			if("stdScaler_2" in typeOfStd):
				print("Starting to cutoff standardized signal between -2 and +2")
				x_train = np.where(x_train>2, 2, x_train)
				x_train = np.where(x_train<-2,-2, x_train)
				x_dev = np.where(x_dev>2, 2, x_dev)
				x_dev = np.where(x_dev<-2, -2, x_dev)
				print("Completed cutoff standardization between -2 and +2")
			print("Finish standard scaling\n")
		elif(typeOfStd == "robustScaler"):
			print("Start robust scaling\n")
			scaler = RobustScaler()
			x_train = scaler.fit_transform(x_train)
			x_dev   = scaler.transform(x_dev)
			print("Finish robust scaling\n")
		elif(typeOfStd=="noScaling"):
			print("No scaling is applied")
			##The x_Train and x_dev needs to be converted to numpy array for plotting
			x_train = x_train.to_numpy()
			x_dev = x_dev.to_numpy()
		#M##full might be standardizing each row separately, can use built in methods for this
		#Melif(typeOfStd == "full"):
		#M	x_train,mean,std = data.fitTransform(x_train.to_numpy())
		#M	fullStats["mean"] = mean
		#M	fullStats["std"] = std
		#M	x_dev = data.transform(x_dev.to_numpy())
		elif(typeOfStd == "onlyMean"):
			print(f"Started {typeOfStd} pre-processing")
			##Only making the mean to 0, keeping the variance from the original signal
			x_train = x_train.sub(x_train.mean(axis=1),axis=0).to_numpy()
			x_dev = x_dev.sub(x_dev.mean(axis=1),axis=0).to_numpy()
			##-----Multiply by 100 or 1000 to get better resolution for amplitude difference---####
			mul=1000
			x_train = np.multiply(x_train, mul)
			x_dev = np.multiply(x_dev, mul)
			##-------------------------------##
			print(f"Finished {typeOfStd} pre-processing")
		elif(typeOfStd=="rowStandard"):
			##Each row is standardized to have 0 mean and 1 stdDev
			x_train = x_train.sub(x_train.mean(axis=1),axis=0).div(x_train.std(axis=1), axis=0).to_numpy()
			x_dev = x_dev.sub(x_dev.mean(axis=1),axis=0).div(x_dev.std(axis=1), axis=0).to_numpy()
		elif(typeOfStd=="newMinMax"):
			##Each colum is first subtracted with its min value, then divided by each columns (max-min)
			##Assumption: We assume that there is very less difference in the traces, so this can work to separate them out.
			##Note: We see that this transformation varies the min and max in dev set. Maybe this could also affect the dev set performance
			x_train_std = x_train.sub(x_train.min(axis=0)).div(x_train.max(axis=0).sub(x_train.min(axis=0))).to_numpy()
			x_dev = x_dev.sub(x_train.min(axis=0)).div(x_train.max(axis=0).sub(x_train.min(axis=0))).to_numpy()
			x_train = x_train_std
		elif(typeOfStd=="minMax"):
		#	##Here each row is normalized between 0 and 1
		#	x_train = x_train.sub(x_train.min(axis=1), axis=0).div(x_train.max(axis=1).sub(x_train.min(axis=1)),axis=0).to_numpy()
		#	x_dev = x_dev.sub(x_dev.min(axis=1), axis=0).div(x_dev.max(axis=1).sub(x_dev.min(axis=1)),axis=0).to_numpy()
		
			##Here each row is normalized between -1 and 1
			x_train = x_train.sub(x_train.min(axis=1), axis=0).div(x_train.max(axis=1).sub(x_train.min(axis=1)),axis=0).mul(2).sub(1).to_numpy()
			x_dev = x_dev.sub(x_dev.min(axis=1), axis=0).div(x_dev.max(axis=1).sub(x_dev.min(axis=1)),axis=0).mul(2).sub(1).to_numpy()
		
		elif(typeOfStd=="logTransform"):
			##Here each row is first sqaured and then natural log is taken, this looks like it eggagerates the small differences within the traces.
			x_train = np.log10(np.power(x_train.to_numpy(),2))
			x_dev = np.log10(np.power(x_dev.to_numpy(),2))
		elif(typeOfStd=="maxAbs"):
			scaler = MaxAbsScaler()
			x_train = scaler.fit_transform(x_train)
			x_dev = scaler.transform(x_dev)
		
		elif(typeOfStd=="stft"):
			C = 200
			noverlap = None 
			nperseg = 256
			x_dev = convert_stft(x_dev.values, noverlap=noverlap, nperseg=nperseg)*C
			x_train = convert_stft(x_train.values, noverlap=noverlap, nperseg=nperseg)*C
		elif(typeOfStd == "lpf" or typeOfStd == "lpf_onlyMean" or typeOfStd == "lpf_std"): ##low pass filter
			fs = 22050.0       # Sample rate, Hz
			##Frequency for cut-off
			cutoff = 5600.0    # Desired cutoff frequency, Hz
			trans_width = 100  # Width of transition from pass band to stop band, Hz
			##numtaps if low will give plots with troughs, need tobe fairly high for a good response
			numtaps = 256     # Size of the FIR filter.
			taps = signal.remez(numtaps, [0, cutoff, cutoff + trans_width, 0.5*fs], [1, 0], Hz=fs)
			x_train = signal.lfilter(taps, [1], x_train)
			x_dev = signal.lfilter(taps, [1], x_dev)
			if(typeOfStd == "lpf_onlyMean"):
				x_train = x_train.to_numpy() - x_train.mean(axis=1).values.reshape(len(x_train),1)
				x_dev = x_dev.to_numpy() - x_dev.mean(axis=1).values.reshape(len(x_dev),1)
			elif(typeOfStd == "lpf_std"):
				scaler = StandardScaler()
				x_train = scaler.fit_transform(x_train)
				x_dev   = scaler.transform(x_dev)
				
		else:
			print("Choose a correct typeOfStd given {typeOfStd}")
			raise NameError(f"incorrect typeOfStd: {typeOfStd}, choose correct value")

		return x_train, x_dev

		
######-----------------------------------------------------------#######
