#!/usr/bin/env python3
import numpy as np


SPEED_THRUST_TABLE = [
    [0.000,   0],
    [0.636,   125],
    [0.990,   250],
    [1.280,   375],
    [1.520,   500],
    [1.733,   625],
    [1.924,   750],
    [2.260,  1000],
    [2.568,  1250],
    [2.840,  1500],
    [3.093,  1750],
    [3.320,  2000],
    [4.830,  4000],
    [5.370,  5000],
]

thrust_table = np.asarray(SPEED_THRUST_TABLE)

speeds = np.array(thrust_table[:,0])
thrusts = np.array(thrust_table[:,1])

def speed_thrust_wamv(speed):
    
    return np.interp(speed, speeds, thrusts)
    

