import os
import sys
sys.path.append(os.getcwd())

import json
import random
import argtyped
import re
from tqdm import tqdm
from typing import Tuple, List
from pathlib import Path

from dataset_process.helpers import load_tsv

class Arguments(argtyped.Arguments, underscore=True):
    sample_path: Path
    template: Path
    caption: Path
    direction: Path
    output: Path
    direction_words: Tuple[List[str], ...] = (
        ["left"],
        ["right"],
        ["upstairs", "up"],
        ["downstairs", "down"],
        ["forward", "straight"],
        ["around"]
    )


def run_insertion(args: Arguments):
    fieldnames = ["instr_id", "sentence"]
    templates = load_tsv(args.template, fieldnames)
    captions = json.load(open(args.caption))
    directions = json.load(open(args.direction))
    sample_paths = json.load(open(args.sample_path))
    # record the nums of the mask and omask for each template
    temps_num = [] # num of mask_omask
    temps_indexs = [[] for i in range(200)] # indexs of tempaltes for each num of mask_omask
    for ind, temp in tqdm(enumerate(templates)):
        instr = temp['sentence']
        index = ind
        instr = re.sub('([.,!?:()])', r' \1', instr)
        words = instr.split(' ')
        mask_num = words.count('[MASK]')
        omask_num = words.count('[OMASK]')
        t_n = str(mask_num)+'_'+str(omask_num)
        if t_n not in temps_num:
            temps_num.append(t_n)
        temps_indexs[temps_num.index(t_n)].append(index)

    new_data = []
    for ind, traj in tqdm(enumerate(sample_paths)):

        cap_item = captions[ind]
        dir_item = directions[ind]

        m_om = str(len(cap_item)) +'_'+ str(len(dir_item))
        if m_om not in temps_num:  #donnot have matched templates
            continue
        else:
            instr = templates[random.choice(temps_indexs[temps_num.index(m_om)])]['sentence']

        instr = re.sub('([.,!?:()])', r' \1', instr)
        words = instr.split(' ')
        mask_num = words.count('[MASK]')
        omask_num = words.count('[OMASK]')

        mask_indexes = None
        if mask_num > 0:
            cap_words = []
            viewpoint_indexes = []
            for i, w in enumerate(cap_item):
                room, obj = w.split(' with ')
                caps = [w,room,obj]
                cap_words.append(random.sample(caps,1)[0])
                viewpoint_indexes.append(i + 1)  # for room/object/ room with object

            mask_indexes = [i for i in range(len(words)) if words[i] == '[MASK]']

            for i, index in enumerate(mask_indexes):
                words[index] = cap_words[i]

        if omask_num > 0:
            omask_indexes = [i for i in range(len(words)) if words[i] == '[OMASK]']
            i = 0
            if mask_indexes is not None:
                for index in omask_indexes:
                    while i < len(mask_indexes) and mask_indexes[i] < index:
                        i += 1
                    if i < len(mask_indexes):
                        curr_view = viewpoint_indexes[i]
                    else:
                        curr_view = viewpoint_indexes[i - 1]
                    if curr_view - 1 >= 0:
                        dir = dir_item[curr_view - 1]
                    else:
                        dir = dir_item[curr_view]
                    if 'around' in dir:
                        dir_w = 'around'
                    elif index - 1 >= 0 and words[index - 1] == 'turn':
                        for w in dir:
                            if w != 'forward':
                                dir_w = w
                                break
                        else:
                            dir_w = random.choice(dir)
                    else:
                        dir_w = random.choice(dir)
                    words[index] = dir_w
            else:
                if omask_num < len(viewpoint_indexes):
                    sample_indexes = random.sample(viewpoint_indexes, omask_num)
                    sample_indexes.sort()
                else:
                    omask_indexes = random.sample(omask_indexes, len(viewpoint_indexes))
                    omask_indexes.sort()
                    sample_indexes = viewpoint_indexes
                for i, index in enumerate(omask_indexes):
                    curr_view = sample_indexes[i]
                    while curr_view > len(dir_item):
                        curr_view -= 1
                    if curr_view - 1 >= 0:
                        dir = dir_item[curr_view - 1]
                    else:
                        dir = dir_item[curr_view]
                    if 'around' in dir:
                        dir_w = 'around'
                    elif index - 1 >= 0 and words[index - 1] == 'turn':
                        for w in dir:
                            if w != 'forward':
                                dir_w = w
                                break
                        else:
                            dir_w = random.choice(dir)
                    else:
                        dir_w = random.choice(dir)
                    words[index] = dir_w
        new_instr = ' '.join(words)
        new_traj = traj.copy()
        new_traj['instructions'].append(new_instr)
        new_traj['path_id'] = new_traj['id']
        new_data.append(new_traj)

    with open(args.output, 'w') as f:
        json.dump(new_data, f, indent=4)


if __name__ == "__main__":
    args = Arguments()
    run_insertion(args)

