import numpy as np

def cartesian_to_polar(polar_state):
    '''Takes in a 2 by 1 vector [theta r].T and transforms it to a 4 by 1 [x y v_x v_y].T'''
    cartesian_state = np.zeros((4,1))
    cartesian_state[0] = polar_state[1]*np.cos(polar_state[0])
    cartesian_state[1] = polar_state[1]*np.sin(polar_state[0])
    # cartesian_state[2] and cartesian_state[3] are 0 as sensor doesn't output v_x and v_y
    return cartesian_state

def jacobian(r, theta):
    return np.array([[np.cos(theta), -r*np.sin(theta)],
                     [np.sin(theta), r*np.cos(theta)],
                     [0, 0],
                     [0, 0]])