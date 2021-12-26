import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import pandas as pd
import numpy as np
import time

def PrepareData(r_dir = "TrainingData/Ready/"):
	# Import our data

	num_of_feat  = 5
	total_column = 7

	test_data = pd.read_csv("{}{}".format(r_dir, "test_data.csv"))


	# Separating the data
	test_data = test_data.to_numpy()

	x_test = test_data[:, 0:num_of_feat]
	y_test = test_data[:, num_of_feat:total_column]

	print ("..Data Prepared")
	return x_test, y_test

def PrepareModel(model_filepath):

	model = tf.keras.models.load_model(model_filepath)

	model.compile(loss='mean_squared_error',
				optimizer='adam',
				metrics=['MSE'])

	print ("..Model Prepared")

	return model


def EvaluateModel(model, x_test, y_test):

	# Evaluate our model
	model.evaluate(x_test, y_test)


def NNModelEvaluate():

	print ("NN Evaluation Start")
	start_time = time.time()

	model_filepath = "SavedModels/annModels_20211223_1100_paper"

	# Prepare data
	x_test, y_test = PrepareData()

	# Prepare model
	model = PrepareModel(model_filepath)

	# Evaluate model
	EvaluateModel(model, x_test, y_test)

	# Done
	end_time = time.time()
	run_time = end_time - start_time

	print("total NN evaluation time: ", "{:.0f} minutes, {:.0f} seconds".format(run_time / 60, run_time % 60))
	print ("NN Training Done")



## Run if this script called
if __name__ == "__main__":
	NNModelEvaluate()