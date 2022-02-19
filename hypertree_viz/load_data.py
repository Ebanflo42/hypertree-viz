import numpy as np
import pickle as pkl
from tqdm import tqdm

print('Loading vertices . . .')
with open('/home/medusa/Projects/hyperbolic-embedder/billeh-coordinates.txt', 'r') as f:
    lines = f.readlines()
    NPTS = len(lines) - 2
    vertices = np.zeros((NPTS, 3), dtype=np.float32)
    perm = np.zeros(NPTS, dtype=np.int32)
    for i, line in tqdm(enumerate(lines[2:])):
        coordstrings = line.split()
        perm[i] = int(coordstrings[0])
        r, phi = float(coordstrings[1]) / \
            13.91, np.pi*float(coordstrings[2])/180
        vertices[perm[i]] = np.array([(5/9)*r*np.cos(phi), r*np.sin(phi), 1])

colors = np.zeros((NPTS, 3), dtype=np.float32)
print('Finding neuron types . . .')
with open('/home/medusa/Projects/billeh_tinkering/neuron_types.pkl', 'rb') as f:
    neuron_types = pkl.load(f)['neuron_type_names']
with open('/home/medusa/Projects/billeh_tinkering/network_dat_original.pkl', 'rb') as f:
    nodes = pkl.load(f)['nodes']
    for typename, node in tqdm(zip(neuron_types, nodes)):
        if typename.startswith('e') and '1' in typename:
            for id in node['ids']:
                colors[id] = np.array([1, 0, 0])
        elif typename.startswith('e') and '23' in typename:
            for id in node['ids']:
                colors[id] = np.array([1, 0.25, 0])
        elif typename.startswith('e') and '4' in typename:
            for id in node['ids']:
                colors[id] = np.array([1, 0.5, 0])
        elif typename.startswith('e') and '5' in typename:
            for id in node['ids']:
                colors[id] = np.array([1, 0.75, 0])
        elif typename.startswith('e') and '6' in typename:
            for id in node['ids']:
                colors[id] = np.array([1, 1, 0])

        elif typename.startswith('i') and '1' in typename:
            for id in node['ids']:
                colors[id] = np.array([0, 0, 1])
        elif typename.startswith('i') and '23' in typename:
            for id in node['ids']:
                colors[id] = np.array([0, 0.25, 1])
        elif typename.startswith('i') and '4' in typename:
            for id in node['ids']:
                colors[id] = np.array([0, 0.5, 1])
        elif typename.startswith('i') and '5' in typename:
            for id in node['ids']:
                colors[id] = np.array([0, 0.75, 1])
        elif typename.startswith('i') and '6' in typename:
            for id in node['ids']:
                colors[id] = np.array([0, 1, 1])
