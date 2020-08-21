import math
from itertools import combinations, product
import xml.etree.ElementTree as ET
from glob import glob
import arff
from collections import namedtuple
import numpy as np
import arff

Minutia = namedtuple('Minutia', ['x', 'y', 'id', 'angle', 'type', 'score'])

def euclidian_distance(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2+(y1-y2)**2)

def max_distance(minutie_list):
    return max(euclidian_distance(m1.x,m1.y,m2.x,m2.y) for m1, m2 in combinations(minutie_list,2))

def get_minutiae_list(mntscore_path):
    tree = ET.parse(mntscore_path)

    type_path = 'SD27_Latent_xml/'+mntscore_path.split('\\')[-1].split('.')[0]+'.xml'
    type_tree = ET.parse(type_path)
    
    rv = []


    type_iter = type_tree.getroot().iter('Minutia')
    for e in tree.getroot().iter('Experiment'):
        t = next(type_iter)
        m = e.find('MissingMinutia') 
        
        rv.append(  Minutia(int(m.attrib['x']), 
                            int(m.attrib['y']), 
                            int(e.attrib['version'][1:]),
                            float(m.attrib['angle']), 
                            t.attrib['Type'],
                            float(e.attrib['score']))
                )
        
    return rv


def ad2pi(alpha, beta):
    if beta > alpha:
        return beta - alpha
    else:
        return beta - alpha + 2*math.pi

def ang(x1, y1, x2, y2):
	deltax = x1 - x2 
	deltay = y1 - y2
	if deltax > 0 and deltay >= 0:
		return np.arctan(deltay/deltax)
	elif deltax > 0 and deltay < 0:
		return np.arctan(deltay/deltax) + 2*math.pi
	elif deltax < 0:
		return np.arctan(deltay/deltax) + math.pi
	elif deltax == 0 and deltay > 0:
		return math.pi/2
	elif deltax == 0 and deltay < 0:
		return 3*math.pi/2

def fix_data_type(row):
    return map(_fix_data_type, row)

def _fix_data_type(data_value):
    try:
        return int(data_value)
    except ValueError:
        try:
            return float(data_value)
        except ValueError:
            return data_value


if __name__ == "__main__":
    max_distance = math.ceil(max(max_distance(get_minutiae_list(filename)) for filename in glob('./diff/*.mntscore')))

    # print(min((len(get_minutiae_list(filename)),filename) for filename in glob('./diff/*.mntscore')))

    data_list =[]
    
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

        #angles
        argsorted_distances = np.argsort(distances, axis=1)

        beta_angles = np.empty((mc,13))
        beta_angles.fill(np.nan)
        alpha_angles = np.empty((mc,13))
        alpha_angles.fill(np.nan)
        extra_angles = np.empty((mc,13))
        extra_angles.fill(np.nan) 
        alpha_neighbors_angles = np.empty((mc,13))
        alpha_neighbors_angles.fill(np.nan)
       

        for pm in range(mc):
            pa = minutiae_list[pm].angle
            px = minutiae_list[pm].x
            py = minutiae_list[pm].y

            for idx, qm in enumerate(argsorted_distances[pm, 1:min_col]):
                beta_angles[pm][idx] = ad2pi( minutiae_list[qm].angle, pa)

                ang_m = ang(px, py, minutiae_list[qm].x, minutiae_list[qm].y)

                alpha_angles[pm][idx] = ad2pi(ang_m, pa)

                alpha_neighbors_angles[pm][idx] = ad2pi(ang_m, minutiae_list[qm].angle)

                extra_angles[pm][idx] = ad2pi(alpha_angles[pm][idx], beta_angles[pm][idx])


            beta_angles[pm][-1] = ad2pi(pa,  minutiae_list[argsorted_distances[pm,-1]].angle)

            ang_m = ang(px, py, minutiae_list[argsorted_distances[pm,-1]].x, minutiae_list[argsorted_distances[pm,-1]].y)

            alpha_angles[pm][-1] = ad2pi(ang_m, pa)

            alpha_neighbors_angles[pm][-1] = ad2pi(ang_m, minutiae_list[argsorted_distances[pm,-1]].angle)

            extra_angles[pm][-1] = ad2pi(alpha_angles[pm][-1], beta_angles[pm][-1])

        

        type_and_score = np.array([(m.type, m.score) for m in minutiae_list])

        fingerprint_data = np.concatenate(( neighbors_count, relative_neighbors_count, neighbors_count_increment, relative_neighbors_count_increment,
                                            nearest_distances, relative_nearest_distances, nearest_distances_increment, relative_nearest_distances_increment,
                                            alpha_angles, alpha_neighbors_angles, beta_angles, extra_angles, 
                                            type_and_score),axis=1)

        data_list.append(fingerprint_data)
    
    data_numpy = np.concatenate(data_list, axis=0)

    data = map(fix_data_type, data_numpy.tolist())

    names = []

    
    names += ('nn{r}'.format(r=r) for r in range(15,max_distance,15))
    names.append('nn{r}'.format(r=max_distance))

    names += ('nn{r}r'.format(r=r) for r in range(15,max_distance,15))
    names.append('nn{r}r'.format(r=max_distance))

    names += ('nn{r1}-nn{r2}'.format(r1=r, r2=r-15) for r in range(30,max_distance,15))
    names.append('nn{r1}-nn{r2}'.format(r1=max_distance, r2=(max_distance//15)*15))

    names += ('nn{r1}r-nn{r2}r'.format(r1=r, r2=r-15) for r in range(30,max_distance,15))
    names.append('nn{r1}r-nn{r2}r'.format(r1=max_distance, r2=(max_distance//15)*15))

    names += ('d{r}'.format(r=r) for r in range(1,13))
    names.append('df')

    names += ('d{r}r'.format(r=r) for r in range(1,13))
    names.append('dfr')

    names += ('d{r1}-d{r2}'.format(r1=r, r2=r-1) for r in range(2,13))
    names.append('df-d12')

    names += ('d{r1}r-d{r2}r'.format(r1=r, r2=r-1) for r in range(2,13))
    names.append('dfr-d12r')

    names += ('alpha{r}'.format(r=r) for r in range(1,13))
    names.append('alphaf')

    names += ('alphan{r}'.format(r=r) for r in range(1,13))
    names.append('alphanf')

    names += ('beta{r}'.format(r=r) for r in range(1,13))
    names.append('betaf')

    names += ('alpha{r}-beta{r}'.format(r=r) for r in range(1,13))
    names.append('alphaf-betaf')

    names += ['type', 'score']

    print(set(data_numpy[:,-2]))

    # arff.dump('data.arff', data, relation='Fingerprint_score_change', names=names)

        