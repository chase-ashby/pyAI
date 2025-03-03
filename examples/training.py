import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

def generate_training_set(num_points):
    ''' 
    Generate simple training data for SLP 
    '''
    x_coordinates = [random.randint(0, 50) for i in range(num_points)]
    y_coordinates = [random.randint(0, 50) for i in range(num_points)]
    training_set = dict()
    for x, y in zip(x_coordinates, y_coordinates):
        if x <= 45-y:
            training_set[(x,y,1)] = 1
        elif x > 45-y:
            training_set[(x,y,1)] = -1
    return training_set