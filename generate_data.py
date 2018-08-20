import numpy as np
import matplotlib.pyplot as plt

def polar_ellipse(theta, semimajor, semiminor):
    denom = ((np.cos(theta)**2)*semiminor**2) + ((np.sin(theta)**2)*semimajor**2)
    return (semimajor*semiminor)/(np.sqrt(denom))

def generate_true_ellipse(num_trials, dt, semimajor, semiminor):
    state = np.zeros((2,num_trials)) # [theta, r].T
    theta = np.linspace(0, 2*np.pi, num_trials)
    
    state[0,0] = theta[0]
    state[1,0] = polar_ellipse(theta[0], semimajor, semiminor)
    for i in range(1, num_trials):
        state[0,i] = theta[i]
        state[1,i] = polar_ellipse(theta[i], semimajor, semiminor)
    return state

def generate_GPS_ellipse(num_trials, dt, semimajor, semiminor, std_dev_r):
    state = np.zeros((2,num_trials)) # [theta, r].T
    theta = np.linspace(0, 2*np.pi, num_trials)
    for i in range(0, num_trials):
        state[0,i] = theta[i]
        state[1,i] = polar_ellipse(theta[i], semimajor, semiminor) + np.random.normal(0, std_dev_r)
    return state

def plotEllipse(num_trials, dt, semimajor, semiminor, std_dev_r):
    state = generate_GPS_ellipse(num_trials, dt, semimajor, semiminor, std_dev_r)
    
    plt.figure()
    plt.polar(state[0],state[1])
    plt.show()

plotEllipse(1000,0.001,200,100, 3)
