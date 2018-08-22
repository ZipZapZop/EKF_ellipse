import numpy as np

def cartesian_to_polar(polar_state):
    '''Takes in a 2 by 1 vector [theta r].T and transforms it to a 4 by 1 [x y v_x v_y].T'''
    cartesian_state = np.zeros((4,1))
    cartesian_state[0] = polar_state[1]*np.cos(polar_state[0])
    cartesian_state[1] = polar_state[1]*np.sin(polar_state[0])
    # cartesian_state[2] and cartesian_state[3] are 0 as sensor doesn't output v_x and v_y
    return cartesian_state

def jacobian(polar_state):
    return np.array([[np.cos(polar_state[0]), -polar_state[1]*np.sin(polar_state[0])],
                     [np.sin(polar_state[0]), polar_state[1]*np.cos(polar_state[0])],
                     [0, 0],
                     [0, 0]])

def new_jacobian(state): # comes in as 4x1 vector - [x, y, v_x, v_y].T
    xsq_plus_ysq = state[0]**2 + state[1]**2
    return np.array([[state[0]/np.sqrt(xsq_plus_ysq), state[1]/np.sqrt(xsq_plus_ysq), 0, 0],
                     [-state[1]/xsq_plus_ysq, -state[0]/xsq_plus_ysq, 0, 0]])