import os
import sys
sys.path.append(os.getcwd())

import csv
import json
import numpy as np
import base64
import torch
import MatterSim
from tqdm import tqdm

from dataset_process.env import R2RBatch


env_actions = {
      'left': (0,-1, 0), # left
      'right': (0, 1, 0), # right
      'up': (0, 0, 1), # up
      'down': (0, 0,-1), # down
      'forward': (1, 0, 0), # forward
      '<end>': (0, 0, 0), # <end>
      '<start>': (0, 0, 0), # <start>
      '<ignore>': (0, 0, 0)  # <ignore>
    }


def take_action(name):
    if type(name) is int:  # Go to the next view
        environment.env.sim.makeAction(name, 0, 0)
    else:  # Adjust
        environment.env.sim.makeAction(*env_actions[name])


VIEWPOINT_SIZE = 36
FEATURE_SIZE = 512
trajectories = json.load(open('data/task/sample_paths.json'))
features = {}
tsv_fieldnames = ['scanId', 'viewpointId', 'image_w', 'image_h', 'vfov', 'features']
with open('data/img_features/CLIP-ViT-B-32-views.tsv', "r") as tsv_in_file:
    reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=tsv_fieldnames)
    for item in reader:
        long_id = item['scanId'] + "_" + item['viewpointId']
        features[long_id] = np.frombuffer(base64.decodestring(item['features'].encode('ascii')),
                                          dtype=np.float32).reshape((VIEWPOINT_SIZE, FEATURE_SIZE))

environment = R2RBatch(features, trajectories)
directions = []
for ind, traj in tqdm(enumerate(trajectories)):
    if ind == 0:
        ob = environment.reset()
    else:
        ob = environment.next()
    direction_traj = []
    for v_ind, viewpoint in enumerate(traj['path']):
        if v_ind != len(traj['path']) - 1:
            for k, candidate in enumerate(ob['candidate']):
                if candidate['viewpointId'] == ob['teacher']:
                    action = k
                    break
            else:  # Stop here
                assert ob['teacher'] == ob['viewpoint']  # The teacher action should be "STAY HERE"
                action = -1

            direction_step = []
            if action != -1:            # -1 is the <stop> action
                select_candidate = ob['candidate'][action]
                src_point = ob['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point) // 12  # The point idx started from 0
                trg_level = (trg_point) // 12

                if src_level < trg_level:
                    direction_step.append('up')
                elif src_level > trg_level:
                    direction_step.append('down')
                else:
                    direction_step.append('forward')

                while src_level < trg_level:  # Tune up
                    take_action('up')
                    src_level += 1
                while src_level > trg_level:  # Tune down
                    take_action('down')
                    src_level -= 1

                curr_point = environment.env.sim.getState().viewIndex
                if trg_point > curr_point:
                    inter = trg_point - curr_point
                    if inter < 2 &inter > 10: #0 1 11
                        direction_step.append('forward')
                    if inter < 6:
                        direction_step.append('right')
                    elif inter > 6:
                        direction_step.append('left')
                    else:
                        direction_step.append('around')

                elif trg_point < curr_point:
                    inter = curr_point - trg_point
                    if inter < 2 & inter > 10:  # 0 1 11
                        direction_step.append('forward')
                    elif inter < 6:
                        direction_step.append('left')
                    elif inter > 6:
                        direction_step.append('right')
                    else:
                        direction_step.append('around')

                while environment.env.sim.getState().viewIndex != trg_point:  # Turn right until the target
                    take_action('right')
                assert select_candidate['viewpointId'] == \
                        environment.env.sim.getState().navigableLocations[select_candidate['idx']].viewpointId
                take_action(select_candidate['idx'])

                ob = environment._get_obs()
            direction_traj.append(direction_step)
    directions.append(direction_traj)

with open('data/task/sample_path_directions.json', 'w') as f:
    json.dump(directions, f, indent=4)

