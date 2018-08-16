import numpy as np
import matplotlib.pyplot as plt

def polar_ellipse(theta, semimajor, semiminor):
    denom = ((np.cos(theta)**2)*semiminor**2) + ((np.sin(theta)**2)*semimajor**2)
    return (semimajor*semiminor)/(np.sqrt(denom))

def generate_true_ellipse(num_trials, dt, semimajor, semiminor):
    r = np.zeros(num_trials) # r
    theta = np.linspace(0, 2*np.pi, num_trials)
    
    
    r[0] = polar_ellipse(theta[0], semimajor, semiminor)
    for i in range(1, num_trials):
        r[i] = polar_ellipse(theta[i], semimajor, semiminor)
    return r, theta

def generate_GPS_ellipse(num_trials, dt, semimajor, semiminor, std_dev_r):
    r = np.zeros(num_trials)
    theta = np.linspace(0, 2*np.pi, num_trials)
    for i in range(0, num_trials):
        r[i] = polar_ellipse(theta[i], semimajor, semiminor) + np.random.normal(0, std_dev_r)
    return r, theta

def plotEllipse(num_trials, dt, semimajor, semiminor, std_dev_r):
    r, theta = generate_GPS_ellipse(num_trials, dt, semimajor, semiminor, std_dev_r)
    
    plt.figure()
    plt.polar(theta,r)
    plt.show()

plotEllipse(1000,0.001,200,100, 3)