import math
from math import cos, sin, tan

import numpy as np
import matplotlib.pyplot as plt

deg2rad = math.pi / 180

## Simulations Param.
dt = 0.01 # Seconds
runtime = 5 # Seconds
index_num = int (runtime / dt)

## System Param.
M = 5    # kg
m = 0.5  # kg
g = 9.81 # m/s2
l = 2    # m
f = 5    # N, constant
y = 0    # m
theta_deg = 0.1 # degree

theta = theta_deg * deg2rad

y_d = 0      # m/s 
theta_d = 0  # m/s

y_dd = 0      # m/s 
theta_dd = 0  # m/s

# F(x) for holding theta
f = g*tan(theta)*(M+m-m*(cos(theta)**2)) + m*g*sin(theta)*cos(theta) - m*(theta_d^theta_d)*l*sin(theta)

## Initialize

t = 0 # Seconds
T = np.zeros(index_num)

Y = np.zeros(index_num)
Y_d = np.zeros(index_num)
Y_dd = np.zeros(index_num)

Theta = np.zeros(index_num)
Theta_d = np.zeros(index_num)
Theta_dd = np.zeros(index_num)

Data = np.zeros([index_num, 3])

## Simulations

for i in range(index_num):
   
    # Update Time
    t = t + dt;
    #disp(t)
    
    # Update Accel.
    y_dd = (f - m*g*sin(theta)*cos(theta) + m*(theta_d*theta_d)*l*sin(theta)) / (M+m-m*(cos(theta)**2))
    theta_dd = (g*sin(theta) - y_dd*cos(theta)) / l
    
    # Update Vel.
    y_d = y_d + y_dd * dt
    theta_d = theta_d + theta_dd * dt
    
    # Update Pos.
    y = y + y_d * dt
    theta = theta + theta_d * dt

    
    # Updating Matrix
    
    T[i] = t
    
    Y[i] = y
    Y_d[i] = y_d
    Y_dd[i] = y_dd
    
    Theta[i] = theta
    Theta_d[i] = theta_d
    Theta_dd[i] = theta_dd
    
    Data[i, 0] = t
    Data[i, 1] = y
    Data[i, 2] = theta


plt.figure(1)
plt.plot(T, Y)
#hold on
#plot(T, Theta);
plt.xlabel('Time [s]')
plt.ylabel('y(t) [m]')
plt.legend( ["y(t)"] );
plt.title('Position (y) vs Time (t)')
#hold off

plt.figure(2)
plt.plot(T, Theta * 180 / math.pi);
#hold on
plt.xlabel('Time [s]')
plt.ylabel('theta [deg]')
plt.legend( ["theta"] );
plt.title('Theta vs Time')
#hold off

print(Data)
plt.show()