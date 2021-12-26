from h5py._hl import files
import tensorflow as tf
import tensorflow.keras as keras

import pandas as pd
import numpy as np
from math import atan2, cos, sin, pi
import os

def NormAngle(x):

    while x < 0:
        x = x + 2*pi

    x = x % (2*pi)

    return x 

def GetFilesNames(filepath = "TrainingData/Test/"):

    return os.listdir(filepath)


def ImportModel(model_filepath = "SavedModels/annModels_20211222_1700_paper"):
    ## Load Model

    model = keras.models.load_model(model_filepath)

    model.compile(loss='mean_squared_error',
                optimizer='adam',
                metrics=['MSE'])

    return model


def ImportData(model, filepath= "TrainingData/Test/penData_000578.csv"):
    testDF = pd.read_csv(filepath)

    test_np = testDF.to_numpy()
    x_te = test_np[:, 0:5]
    y_te = test_np[:, 5:7]

    y_pred = model.predict(x_te)

    return x_te, y_te, y_pred


def ManualPrediction(x_te, y_te, y_pred, model):
    ## Manual Prediction
    # With manual theta

    dt = 0.01
    theta_np = np.zeros((y_te.shape[0], 1))
    theta_dt = np.zeros((y_te.shape[0], 1))
    theta = NormAngle(atan2(x_te[0, 1], x_te[0, 2]))
    y_predi = y_pred.copy()

    y_predi[0]  = model(x_te[0, :].reshape([1, 5]), training= False)[0]
    y_predi[0][0] = x_te[0][3] + y_predi[0][0]
    y_predi[0][1] = x_te[0][4] + y_predi[0][1]

    theta_np[0, 0] = theta
    theta_dt[0, 0] = theta
    theta = NormAngle( theta + y_predi[0][1] * dt )

    for i in range(1, y_te.shape[0]):    

        y_predi[i]  = model(np.concatenate( (x_te[i, 0], [sin(theta), cos(theta)] ,y_predi[i-1]), axis=None ).reshape(1, 5), training= False)[0]
        y_predi[i][0] = y_predi[i-1][0] + y_predi[i][0]
        y_predi[i][1] = y_predi[i-1][1] + y_predi[i][1]    

        theta_np[i, 0] = theta
        theta_dt[i, 0] = NormAngle(atan2(x_te[i, 1], x_te[i, 2]))
        #theta = NormAngle( theta + y_predi[i][1] * dt )
        theta = theta + y_predi[i][1] * dt

    theta_dev = abs (theta_dt[2999] - theta_np[2999])
    yDot_dev = abs (x_te[2999, 3] - y_predi[2999, 0])
    thetaDot_dev = abs (x_te[2999, 4] - y_predi[2999, 1])

    return theta_dev, yDot_dev, thetaDot_dev



def SimulateNN():

    # GetFiles
    files_names = GetFilesNames()
    print ("Files Name Obtained")

    # Import Models
    model = ImportModel("SavedModels/annModels_20211222_1700_paper")
    print ("Model is Ready")

    # Creating DataFrames
    SimulationDF = pd.DataFrame(files_names, columns= ["Files Names"])
    SimulationDF["Theta Dev (rad)"] = 999
    SimulationDF["yDot Dev (m/s)"] = 999
    SimulationDF["ThetaDot Dev (rad/s)"] = 999

    print ("Begin Simulations")
    # Do Simulations
    for i in range( len (files_names) ):

        #Import Datas
        x_te, y_te, y_pred = ImportData (model, "TrainingData/Test/{}".format(files_names[i]))

        #Simulate to calculate deviation 
        theta_dev, yDot_dev, thetaDot_dev = ManualPrediction (x_te, y_te, y_pred, model)

        #Assign Values
        SimulationDF.loc[i, "Theta Dev (rad)"] = theta_dev
        SimulationDF.loc[i, "yDot Dev (m/s)"] = yDot_dev
        SimulationDF.loc[i, "ThetaDot Dev (rad/s)"] = thetaDot_dev
        print ("..i = {}/{}".format(i, len(files_names)))

    # Save CSV
    SimulationDF.to_csv("TrainingData/Simulated_30.csv")
    print ("File Saved")



if __name__ == "__main__":
    SimulateNN()
