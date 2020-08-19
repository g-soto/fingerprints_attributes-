import math
from itertools import combinations
import xml.etree.ElementTree as ET
from glob import glob


def euclidian_distance(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2+(y1-y2)**2)

def max_distance(minutie_list):
    return max(euclidian_distance(x1,y1,x2,y2) for (x1,y1), (x2,y2) in combinations(minutie_list,2))
    
def get_minutiae_list(mntscore_path):
    tree = ET.parse(mntscore_path)
    return ((int(mm.attrib['x']), int(mm.attrib['y'])) for mm in tree.getroot().iter('MissingMinutia'))


if __name__ == "__main__":
    print(max(max_distance(get_minutiae_list(filename)) for filename in glob('./diff/*.mntscore')))
