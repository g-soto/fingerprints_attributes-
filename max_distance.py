import math
from itertools import combinations, product
import xml.etree.ElementTree as ET
from glob import glob
import arff
from collections import namedtuple
import numpy as np
import arff

Minutia = namedtuple('Minutia', ['x', 'y', 'id'])

def euclidian_distance(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2+(y1-y2)**2)

def max_distance(minutie_list):
    return max((euclidian_distance(m1.x,m1.y,m2.x,m2.y), (m1.id,m2.id)) for m1, m2 in combinations(minutie_list,2))

def get_minutiae_list(mntscore_path):
    tree = ET.parse(mntscore_path)
    rv = []

    for e in tree.getroot().iter('Experiment'):
        m = e.find('MissingMinutia')
        rv.append(Minutia(int(m.attrib['x']), int(m.attrib['y']), int(e.attrib['version'][1:])))
        
    return rv


if __name__ == "__main__":
    max_distance = math.ceil(max((max_distance(get_minutiae_list(filename)),filename) for filename in glob('./diff/*.mntscore'))[0][0])

    # print(min((len(get_minutiae_list(filename)),filename) for filename in glob('./diff/*.mntscore')))
    
    rn = math.floor(max_distance/15+1)
    for filename in glob('./diff/*.mntscore'):
        minutiae_list = get_minutiae_list(filename)
        mc = len(minutiae_list)
        distances = np.zeros((mc,mc))

        for m1, m2 in combinations(minutiae_list,2):
            distance = euclidian_distance(m1.x,m1.y,m2.x,m2.y)
            distances[m1.id, m2.id] = distances[m2.id, m1.id] = distance

        #Total number of nearest minutiae in every possible radius from 15 pixels to N(607), with a step of 15 pixels.
        neighbors_count = np.ones((mc,rn))*(mc-1)
        for idx, radius in enumerate(range(15,max_distance,15)):
            neighbors = distances<=radius
            neighbors_count[:,idx] = neighbors.sum(axis=1)-1

        
        #Relative number (respect to max num on nieghbors) of nearest minutiae in every possible radius from 15 pixels to N(607), with a step of 15 pixels.
        relative_neighbors_count = neighbors_count/(mc-1)

        #Absolute increment
        neighbors_count_increment = np.diff(neighbors_count, axis=1)

        #relative increment
        relative_neighbors_count_increment = np.diff(relative_neighbors_count, axis=1)

        #distance (in pixels) to the 12 nearest minutiae in the same fingerprint. and the farest one
        sorted_distances = np.sort(distances, axis=1)

        nearest_distances = np.empty((mc, 13))
        nearest_distances.fill(np.nan)

        min_col = min(13,mc) #amount of cols with non nan

        nearest_distances[:,:min_col-1] = sorted_distances[:,1:min_col]

        nearest_distances[:,-1] = sorted_distances[:,-1]
        
        #relative distance (in pixels) (respect to the midle minutia) to the 12 nearest minutiae in the same fingerprint. and the farest one 
        relative_nearest_distances = nearest_distances/nearest_distances[:,(min_col//2)-1:(min_col//2)]

        #Absolute increment
        nearest_distances_increment = np.empty((mc,12))
        nearest_distances_increment[:,:-1] = np.diff(nearest_distances[:,:-1], axis=1)
        nearest_distances_increment[:,-1] = nearest_distances[:,-1] - nearest_distances[:,min_col-2]

        #relative increment
        relative_nearest_distances_increment = np.empty((mc,12))
        relative_nearest_distances_increment[:,:-1] = np.diff(relative_nearest_distances[:,:-1], axis=1)
        relative_nearest_distances_increment[:,-1] = relative_nearest_distances[:,-1] - relative_nearest_distances[:,min_col-2]