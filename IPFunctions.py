import numpy as np
import pandas as pd
import random
import math
from math import cos, sin, tan


def AngleRoll(angle):
    if angle < 0:
        return angle + (2 * math.pi)
    else:
        return angle % (2 * math.pi)

def SimulatePendulum(filename = "ourData.csv"):
    
    deg2rad = math.pi / 180

    ## Simulations Param.
    dt = 0.01 # Seconds
    runtime = 30 # Seconds
    index_num = int (runtime / dt)

    ## System Param.
    M = 5    # kg
    m = 0.5  # kg
    g = 9.81 # m/s2
    l = 2    # m
    y = 0    # m

    ## Initial Param
    theta = deg2rad * random.randrange(0, 360) # degree
    f = random.randrange(-100, 100)            # N

    y_d = 0      # m/s 
    theta_d = 0  # m/s

    y_dd = 0      # m/s 
    theta_dd = 0  # m/s

    # F(x) for holding theta
    #f = g*tan(theta)*(M+m-m*(cos(theta)**2)) + m*g*sin(theta)*cos(theta) - m*(theta_d^theta_d)*l*sin(theta)
    
    ## Initialize

    t = 0 # Seconds
    Data = np.zeros([index_num, 9])

    ## Simulations

    for i in range(index_num):

        # Colect Input features
        Data[i, 0] = f
        Data[i, 1] = sin(theta)
        Data[i, 2] = cos(theta)
        Data[i, 3] = y_d
        Data[i, 4] = theta_d
        Data[i, 5] = theta_d*theta_d
        Data[i, 6] = cos(theta)**2
       
        # Update Time
        t = t + dt
        #disp(t)
        
        # Update Accel.
        y_dd = (f - m*g*sin(theta)*cos(theta) + m*(theta_d*theta_d)*l*sin(theta)) / (M+m-m*(cos(theta)**2))
        theta_dd = (g*sin(theta) - y_dd*cos(theta)) / l
        
        # Update Vel.
        y_d = y_d + y_dd * dt
        theta_d = theta_d + theta_dd * dt
        
        # Update Pos.
        y = y + y_d * dt
        theta = AngleRoll (theta + theta_d * dt)

        # Colect output data
        
        #Data[i, 5] = sin(theta)
        #Data[i, 6] = cos(theta)
        Data[i, 7] = y_dd * dt
        Data[i, 8] = theta_dd * dt

    ## Save Into CSV
    #DataDF = pd.DataFrame(data=Data, columns=["F", "sin_in", "cos_in", "yDot_in", "thetaD_in", "sin_o", "cos_o", "yDot_del_o", "thetaD_del_o"])
    #DataDF = pd.DataFrame(data=Data, columns=["F", "sin_in", "cos_in", "yDot_in", "thetaD_in", "yDot_del_o", "thetaD_del_o"])
    DataDF = pd.DataFrame(data=Data, columns=["F", "sin_in", "cos_in", "yDot_in", "thetaD_in", "thetaDD_in", "cos2_in", "yDot_del_o", "thetaD_del_o"])
    DataDF.head()

    DataDF.to_csv(filename, index= False)


## Run the script if main
if __name__ == '__main__':
    SimulatePendulum(filename = "ourData.csv")