''' Batched Room-to-Room navigation environment '''
import sys
sys.path.append('/home/lilingling/Matterport3DSimulator/build')

import json
import MatterSim
import csv
import numpy as np
import math
import networkx as nx


csv.field_size_limit(sys.maxsize)


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


class EnvBatch():
    ''' A simple wrapper for a batch of MatterSim environments,
        using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store=None):
        """
        1. Load pretrained image feature
        2. Init the Simulator.
        :param feature_store: The name of file stored the feature.
        :param batch_size:  Used to create the simulator list.
        """
        if feature_store:
            if type(feature_store) is dict:     # A silly way to avoid multiple reading
                self.features = feature_store
                self.image_w = 640
                self.image_h = 480
                self.vfov = 60
                self.feature_size = next(iter(self.features.values())).shape[-1]
                print('The feature size is %d' % self.feature_size)
        else:
            print('    Image features not provided - in testing mode')
            self.features = None
            self.image_w = 640
            self.image_h = 480
            self.vfov = 60
        sim = MatterSim.Simulator()
        sim.setRenderingEnabled(False)
        sim.setDiscretizedViewingAngles(True)   # Set increment/decrement to 30 degree. (otherwise by radians)
        sim.setCameraResolution(self.image_w, self.image_h)
        sim.setCameraVFOV(math.radians(self.vfov))
        sim.init()
        self.sim = sim

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId

    def newEpisodes(self, scanId, viewpointId, heading):
        self.sim.newEpisode(scanId, viewpointId, heading, 0)

    def getStates(self):
        """
        Get list of states augmented with precomputed image features. rgb field will be empty.
        Agent's current view [0-35] (set only when viewing angles are discretized)
            [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
        :return: [ ((30, 2048), sim_state) ] * batch_size
        """
        state = self.sim.getState()

        long_id = self._make_id(state.scanId, state.location.viewpointId)
        if self.features:
            feature = self.features[long_id]
            return (feature, state)
        else:
            return (None, state)

    def makeActions(self, actions):
        ''' Take an action using the full state dependent action interface (with batched input).
            Every action element should be an (index, heading, elevation) tuple. '''
        for i, (index, heading, elevation) in enumerate(actions):
            self.sim.makeAction(index, heading, elevation)


class R2RBatch():
    ''' Implements the Room to Room navigation task, using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store, data):
        self.env = EnvBatch(feature_store=feature_store)
        if feature_store:
            self.feature_size = self.env.feature_size
        else:
            self.feature_size = 2048
        self.data = data
        scans = []
        for item in data:
            scans.append(item['scan'])

        self.scans = set(scans)
        self.sim = self.new_simulator()
        self.ix = 0

        self._load_nav_graphs()

        self.buffered_state_dict = {}

    def new_simulator(self):
        # Simulator image parameters
        WIDTH = 640
        HEIGHT = 480
        VFOV = 60

        sim = MatterSim.Simulator()
        sim.setRenderingEnabled(False)
        sim.setCameraResolution(WIDTH, HEIGHT)
        sim.setCameraVFOV(math.radians(VFOV))
        sim.setDiscretizedViewingAngles(True)
        sim.init()

        return sim

    def size(self):
        return len(self.data)

    def _load_nav_graphs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        :return: None
        """
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.scans)
        self.paths = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _shortest_path_action(self, state, goalViewpointId):
        ''' Determine next action on the shortest path to goal, for supervised training. '''
        if state.location.viewpointId == goalViewpointId:
            return goalViewpointId      # Just stop here
        path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        return nextViewpointId

    def make_candidate(self, feature, scanId, viewpointId, viewId):
        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)
        base_heading = (viewId % 12) * math.radians(30)
        adj_dict = {}
        long_id = "%s_%s" % (scanId, viewpointId)
        if long_id not in self.buffered_state_dict:
            for ix in range(36):
                if ix == 0:
                    self.sim.newEpisode(scanId, viewpointId, 0, math.radians(-30))
                elif ix % 12 == 0:
                    self.sim.makeAction(0, 1.0, 1.0)
                else:
                    self.sim.makeAction(0, 1.0, 0)

                state = self.sim.getState()
                assert state.viewIndex == ix

                # Heading and elevation for the viewpoint center
                heading = state.heading - base_heading
                elevation = state.elevation

                visual_feat = feature[ix]

                # get adjacent locations
                for j, loc in enumerate(state.navigableLocations[1:]):
                    # if a loc is visible from multiple view, use the closest
                    # view (in angular distance) as its representation
                    distance = _loc_distance(loc)

                    # Heading and elevation for for the loc
                    loc_heading = heading + loc.rel_heading
                    loc_elevation = elevation + loc.rel_elevation
                    if (loc.viewpointId not in adj_dict or
                            distance < adj_dict[loc.viewpointId]['distance']):
                        adj_dict[loc.viewpointId] = {
                            'heading': loc_heading,
                            'elevation': loc_elevation,
                            "normalized_heading": state.heading + loc.rel_heading,
                            'scanId':scanId,
                            'viewpointId': loc.viewpointId, # Next viewpoint id
                            'pointId': ix,
                            'distance': distance,
                            'idx': j + 1,
                            # 'feature': np.concatenate((visual_feat, angle_feat), -1)
                            'feature': visual_feat
                        }
            candidate = list(adj_dict.values())
            self.buffered_state_dict[long_id] = [
                {key: c[key]
                 for key in
                    ['normalized_heading', 'elevation', 'scanId', 'viewpointId',
                     'pointId', 'idx']}
                for c in candidate
            ]
            return candidate
        else:
            candidate = self.buffered_state_dict[long_id]
            candidate_new = []
            for c in candidate:
                c_new = c.copy()
                ix = c_new['pointId']
                normalized_heading = c_new['normalized_heading']
                visual_feat = feature[ix]
                loc_heading = normalized_heading - base_heading
                c_new['heading'] = loc_heading
                c_new['feature'] = visual_feat
                c_new.pop('normalized_heading')
                candidate_new.append(c_new)
            return candidate_new

    def _get_obs(self):
        # for i, (feature, state) in enumerate(self.env.getStates()):
        (feature, state) = self.env.getStates()
        item = self.data[self.ix]
        base_view_id = state.viewIndex

        if feature is None:
            feature = np.zeros((36, 2048))

        # Full features
        candidate = self.make_candidate(feature, state.scanId, state.location.viewpointId, state.viewIndex)
        # [visual_feature, angle_feature] for views
        # feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)

        ob = {
            'scan' : state.scanId,
            'viewpoint' : state.location.viewpointId,
            'viewIndex' : state.viewIndex,
            'heading' : state.heading,
            'elevation' : state.elevation,
            'feature' : feature,
            'candidate': candidate,
            'navigableLocations' : state.navigableLocations,
            'teacher' : self._shortest_path_action(state, item['path'][-1]),
            'gt_path' : item['path'],
        }
        # A2C reward. The negative distance between the state and the final state
        ob['distance'] = self.distances[state.scanId][state.location.viewpointId][item['path'][-1]]
        return ob

    def step(self, actions):
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(actions)
        return self._get_obs()

    def next(self):
        self.ix += 1
        scanId = self.data[self.ix]['scan']
        viewpointId = self.data[self.ix]['path'][0]
        heading = float(self.data[self.ix]['heading'])
        self.env.newEpisodes(scanId, viewpointId, heading)
        return self._get_obs()

    def reset(self):
        scanId = self.data[self.ix]['scan']
        viewpointId = self.data[self.ix]['path'][0]
        heading = float(self.data[self.ix]['heading'])
        self.env.newEpisodes(scanId, viewpointId, heading)
        return self._get_obs()

