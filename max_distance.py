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
    return max(euclidian_distance(m1.x,m1.y,m2.x,m2.y) for m1, m2 in combinations(minutie_list,2))

def get_minutiae_list(mntscore_path):
    tree = ET.parse(mntscore_path)
    rv = []

    for e in tree.getroot().iter('Experiment'):
        m = e.find('MissingMinutia')
        rv.append(Minutia(int(m.attrib['x']), int(m.attrib['y']), int(e.attrib['version'][1:])))
        
    return rv


if __name__ == "__main__":
    max_distance = math.ceil(max(max_distance(get_minutiae_list(filename)) for filename in glob('./diff/*.mntscore')))

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
        neighbors_count_increment = np.diff(neighbors_count)

        #relative increment
        relative_neighbors_count_increment = np.diff(relative_neighbors_count)

        #distance (in pixels) to the 12 nearest minutiae in the same fingerprint. and the farest one
        sorted_distances = np.sort(distances, axis=1)

        nearest_distances = np.empty((mc, 13))
        nearest_distances.fill(np.nan)

        nearest_distances[:,:min(12,sorted_distances.shape[1]-1)] = sorted_distances[:,1:min(13,sorted_distances.shape[1])]

        if sorted_distances.shape[1] < 12:
            print(nearest_distances)
            break
        
