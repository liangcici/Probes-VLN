import json
import numpy as np
import networkx as nx
import math
import random
import tqdm


scans = []
with open('data/task/scenes_train.txt', 'r') as f:
    content = f.readlines()
    for con in content:
        scans.append(con.strip())


def load_nav_graphs(scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    graphs = {}
    for scan in scans:
        with open('data/connectivity/%s_connectivity.json' % scan) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i,item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                    item['pose'][7], item['pose'][11]])
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs


distances = {}
graphs = load_nav_graphs(scans)

trajectory_len = 8
error_th = 10
trajectory_num_per_house = 10000

path_data = []
path_id = 0
graph_ind = 0
for scan, G in graphs.items(): # compute all shortest paths
    nodes = dict(G.nodes())
    source_nodes = list(nodes.keys())
    node_len = len(source_nodes)

    for i in tqdm.tqdm(range(trajectory_num_per_house)):
        error_cnt = 0
        while True:
            idx = np.random.randint(node_len)
            target_node = source_nodes[idx]
            node_idx = np.random.randint(node_len)
            source_node = source_nodes[node_idx]
            if idx == node_idx:
                continue
            try:
                traj_len = nx.dijkstra_path_length(G, source_node, target_node)
            except Exception as expt:
                if "not reachable" in str(expt):
                    pass
                continue
            if traj_len <= trajectory_len:
                traj = nx.dijkstra_path(G, source_node, target_node)
                assert traj[-1] == target_node
                obj = {}
                obj['scan'] = scan
                obj['id'] = path_id
                path_id += 1
                obj['path'] = traj
                obj['heading'] = random.random() * 2 * math.pi    # 0 - 2pi
                obj['instructions'] = []
                path_data.append(obj)
                break
            else:
                error_cnt += 1
                if error_cnt > error_th:
                    break

    print('finish house {}'.format(graph_ind))
    graph_ind += 1

with open('data/task/sample_paths.json', 'w') as f:
    json.dump(path_data, f)


