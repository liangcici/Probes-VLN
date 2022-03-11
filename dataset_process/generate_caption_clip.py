import os
import sys
sys.path.append(os.getcwd())

import csv
import json
import clip
import numpy as np
import base64
import torch
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


areas = ['office', 'lounge', 'family room', 'entry way', 'dining room', 'living room', 'stairs', 'kitchen',
         'porch', 'bathroom', 'bedroom', 'hallway']
objects = ['wall', 'floor', 'chair', 'door', 'table', 'picture', 'cabinet', 'cushion', 'window', 'sofa', 'bed',
           'curtain', 'chest of drawers', 'plant', 'sink', 'stairs', 'ceiling', 'toilet', 'stool', 'towel',
           'mirror', 'tv monitor', 'shower', 'column', 'bathtub', 'counter', 'fireplace', 'lighting', 'beam',
           'railing', 'shelving', 'blinds', 'gym equipment', 'seating', 'board panel', 'furniture',
           'appliances', 'clothes']
VIEWPOINT_SIZE = 36
FEATURE_SIZE = 512
TOP_K = 1
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
area_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in areas]).to(device)
object_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in objects]).to(device)
with torch.no_grad():
    area_text_features = model.encode_text(area_inputs)
    object_text_features = model.encode_text(object_inputs)

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
noun_phases = []
candidate_infos = []
for ind, traj in tqdm(enumerate(trajectories)):
    if ind == 0:
        ob = environment.reset()
    else:
        ob = environment.next()
    noun_item = []
    info_item = []
    for v_ind, viewpoint in enumerate(traj['path']):
        if v_ind != len(traj['path']) - 1:
            for k, candidate in enumerate(ob['candidate']):
                if candidate['viewpointId'] == ob['teacher']:
                    action = k
                    break
            else:  # Stop here
                assert ob['teacher'] == ob['viewpoint']  # The teacher action should be "STAY HERE"
                action = -1

            if action != -1:            # -1 is the <stop> action
                select_candidate = ob['candidate'][action]
                src_point = ob['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point) // 12  # The point idx started from 0
                trg_level = (trg_point) // 12
                while src_level < trg_level:  # Tune up
                    take_action('up')
                    src_level += 1
                while src_level > trg_level:  # Tune down
                    take_action('down')
                    src_level -= 1
                while environment.env.sim.getState().viewIndex != trg_point:  # Turn right until the target
                    take_action('right')
                assert select_candidate['viewpointId'] == \
                       environment.env.sim.getState().navigableLocations[select_candidate['idx']].viewpointId
                take_action(select_candidate['idx'])

                cand_feat = torch.from_numpy(select_candidate['feature']).to(device).half()
                cand_feat /= cand_feat.norm(dim=-1, keepdim=True)
                area_text_features /= area_text_features.norm(dim=-1, keepdim=True)
                object_text_features /= object_text_features.norm(dim=-1, keepdim=True)
                area_similarity = (100.0 * cand_feat @ area_text_features.T).softmax(dim=-1)
                object_similarity = (100.0 * cand_feat @ object_text_features.T).softmax(dim=-1)
                area_pred = areas[area_similarity.topk(TOP_K)[1]]
                object_pred = objects[object_similarity.topk(TOP_K)[1]]
                noun_item.append(area_pred + ' with ' + object_pred)

                info = {}
                info['scanId'] = select_candidate['scanId']
                info['viewpointId'] = select_candidate['viewpointId']
                info['pointId'] = select_candidate['pointId']
                info['score'] = [area_similarity.topk(1)[0].item(), object_similarity.topk(1)[0].item()]
                info_item.append(info)

                ob = environment._get_obs()
    noun_phases.append(noun_item)
    candidate_infos.append(info_item)

with open('data/task/sample_path_candidate_info.json', 'w') as f:
    json.dump(candidate_infos, f, indent=4)
with open('data/task/sample_path_predict_noun_phases.json', 'w') as f:
    json.dump(noun_phases, f, indent=4)

