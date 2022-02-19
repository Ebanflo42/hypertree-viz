import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from tqdm import tqdm

with open('/home/medusa/Projects/billeh_tinkering/network_dat_original.pkl', 'rb') as f:
    net_data = pkl.load(f)

n_nodes = 0
for node in net_data['nodes']:
    n_nodes += len(node['ids'])

degrees = np.zeros(n_nodes)
for edge in tqdm(net_data['edges']):
    degrees[edge['source']] += 1
    degrees[edge['target']] += 1
print(degrees)
logbins = np.geomspace(degrees.min(), degrees.max(), 20)
plt.hist(degrees, bins=logbins)
plt.xscale('log')
plt.yscale('log')
plt.show()