import os
import sys
sys.path.append(os.getcwd())

import ast
import argtyped
from typing import List, Any, Tuple, Dict
from pathlib import Path

from dataset_process.helpers import load_tsv, save_tsv


class Arguments(argtyped.Arguments, underscore=True):
    source: Path
    output: Path
    direction_words: Tuple[str, ...] = (
        "left",
        "right",
        "upstairs", "up",
        "downstairs", "down",
        "forward", "straight",
        "around",
    )
    noise_words: Tuple[str, ...] = (
        "veer",
        "exit",
        "go",
        "stop",
        ".",
    )


def mask_noun_phrases(sentences: List[Dict[str, Any]], args: Arguments):
    templates = []
    for item in sentences:
        instr = item['sentence']
        noun_phrases = ast.literal_eval(item['noun_phrases'])
        noun_phrases.sort(key=lambda x: len(x), reverse=True)
        for phrase in noun_phrases:
            is_direction = False
            for dir_w in args.direction_words:
                # 'a left', 'the left'
                if dir_w in noun_phrases and len(noun_phrases) <= 2:
                    is_direction = True
                    break
            if is_direction:
                instr = instr.replace(phrase, '[OMASK]')
            elif phrase not in args.noise_words:
                instr = instr.replace(phrase, '[MASK]')
            instr = instr.replace('\n', ' ')
            instr = instr.replace('\r', '')

            instr_words = instr.split(' ')
            new_instr_words = []
            for w in instr_words:
                if '[MASK]' in w:
                    if w not in ['[MASK]', '[MASK].', '[MASK],', '[MASK]!']:
                        new_instr_words.append('[MASK]')
                    else:
                        new_instr_words.append(w)
                elif '[OMASK]' in w:
                    if w not in ['[OMASK]', '[OMASK].', '[OMASK],', '[OMASK]!']:
                        new_instr_words.append('[OMASK]')
                    else:
                        new_instr_words.append(w)
                else:
                    new_instr_words.append(w)
            instr = ' '.join(new_instr_words)

        templates.append({'instr_id': item['instr_id'], 'sentence': instr})
    return templates


def mask_directions(sentences: List[Dict[str, Any]], args: Arguments):
    templates = []
    for item in sentences:
        instr = item['sentence']
        for word in args.direction_words:
            instr = instr.replace(word, '[OMASK]')

        instr_words = instr.split(' ')
        new_instr_words = []
        for w in instr_words:
            if '[OMASK]' in w:
                if w not in ['[OMASK]', '[OMASK].', '[OMASK],', '[OMASK]!']:
                    new_instr_words.append('[OMASK]')
                else:
                    new_instr_words.append(w)
            else:
                new_instr_words.append(w)
        instr = ' '.join(new_instr_words)

        templates.append({'instr_id': item['instr_id'], 'sentence': instr})
    return templates


def run_mask_words(args: Arguments):
    fieldnames = ["instr_id", "sentence", "noun_phrases"]
    sentences = load_tsv(args.source, fieldnames)

    templates = mask_noun_phrases(sentences, args)
    templates = mask_directions(templates, args)

    template_fieldnames = ["instr_id", "sentence"]
    save_tsv(templates, args.output, template_fieldnames)


if __name__ == "__main__":
    args = Arguments()
    run_mask_words(args)

