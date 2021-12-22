import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import pandas as pd
import numpy as np
import time

def PrepareData(r_dir = "TrainingData/Ready/"):
	# Import our data

	num_of_feat  = 7
	total_column = 9

	train_data = pd.read_csv("{}{}".format(r_dir, "train_data.csv"))
	eval_data = pd.read_csv("{}{}".format(r_dir, "eval_data.csv"))
	test_data = pd.read_csv("{}{}".format(r_dir, "test_data.csv"))


	# Separating the data
	train_data = train_data.to_numpy()
	eval_data = eval_data.to_numpy()
	test_data = test_data.to_numpy()

	x_train = train_data[:, 0:num_of_feat]
	y_train = train_data[:, num_of_feat:total_column]

	x_eval = eval_data[:, 0:num_of_feat]
	y_eval = eval_data[:, num_of_feat:total_column]

	x_test = test_data[:, 0:num_of_feat]
	y_test = test_data[:, num_of_feat:total_column]

	print ("..Data Prepared")
	return x_train, y_train, x_eval, y_eval, x_test, y_test

def PrepareModel():
	model = Sequential()

	## FIRST SET OF LAYERS

	# FEATURE LAYER
	#model.add(Dense(7)) No Need For This Layer


	# 10 NEURONS EACH IN DENSE HIDDEN LAYER
	model.add(Dense(50, activation='relu'))

	model.add(Dense(50, activation='relu'))

	model.add(Dense(50, activation='relu'))

	model.add(Dense(30, activation='relu'))

	# LAST LAYER IS THE OUTPUT LAYER
	model.add(Dense(2))


	model.compile(loss='mean_squared_error',
	              optimizer='adam',
	              metrics=['MSE'])

	print ("..Model Prepared")

	return model

def TrainModel(model, x_train, y_train, x_eval, y_eval):

	# Setting up earlyStopping
	early_stop = EarlyStopping(monitor='val_loss',patience=3)

	# Fit our model
	model.fit(x_train,y_train,epochs=15,validation_data=(x_eval,y_eval),callbacks=[early_stop])

	print ("..Training Complete")

def EvaluateModel(model, x_test, y_test):

	# Evaluate our model
	model.evaluate(x_test, y_test)

def SaveModel(model, model_filepath):

	model.save(model_filepath)

	print("..Model Saved")


def NNModelTrain():

	print ("NN Training Start")
	start_time = time.time()

	save_filepath = "SavedModels/annModels_20211214_0100"

	# Prepare data
	x_train, y_train, x_eval, y_eval, x_test, y_test = PrepareData()

	# Prepare model
	model = PrepareModel()

	# Train Our Model
	TrainModel(model, x_train, y_train, x_eval, y_eval)

	# Evaluate model
	EvaluateModel(model, x_test, y_test)

	# Save Model
	SaveModel(model, save_filepath)

	# Done
	end_time = time.time()
	run_time = end_time - start_time

	print("total NN training time: ", "{:.0f} minutes, {:.0f} seconds".format(run_time / 60, run_time % 60))
	print ("NN Training Done")



## Run if this script called
if __name__ == "__main__":
	NNModelTrain()