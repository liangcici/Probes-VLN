import os
import sys
sys.path.append(os.getcwd())

from typing import List, Dict, Tuple, Any
import json
import re
import csv
import string
import math
import tarfile
import types
from pathlib import Path
import random
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase, AutoTokenizer
from allennlp.predictors.predictor import Predictor
from allennlp.common.util import JsonDict
from allennlp.data import Instance
import argtyped
from dataset_process.helpers import (
    download_file,
    save_txt,
    load_txt,
    load_json,
    save_json,
    load_tsv,
    save_tsv,
)

random.seed(0)


class Arguments(argtyped.Arguments, underscore=True):
    source: Path
    output: Path
    noun_phrases: Path = Path("noun_phrases.json")
    cache_dir: Path = Path.home() / ".cache" / "vln"
    categories: Path = Path("categories.txt")
    matterport: Path = Path("matterport_categories.tsv")
    places365: Path = Path("places365_categories.tsv")
    parser: Path = Path.home() / ".allennlp" / "elmo"
    # forbidden_words: Tuple[str, ...] =("turn",)
    forbidden_words: Tuple[str, ...] =("turn", "you", "head")
    min_tokens: int = 1
    max_tokens: int = 5
    max_instr_length: int = 200
    batch_size: int = 100
    start: int = 0
    num_splits: int = 1
    num_workers: int = 1


def _json_to_instance(self, json_dict: JsonDict) -> Instance:
    """
    Override the function from ConstituencyParserPredictor
    """
    spacy_tokens = self._tokenizer.tokenize(json_dict["sentence"])
    spacy_tokens = spacy_tokens[: self.max_length]
    sentence_text = [token.text for token in spacy_tokens]
    pos_tags = [token.tag_ for token in spacy_tokens]
    return self._dataset_reader.text_to_instance(sentence_text, pos_tags)


def clean_sentence(stc: str) -> str:
    return stc.strip(". ,\n\r").lower()


def create_token(
    tree: Dict,
    tokenizer: PreTrainedTokenizerBase,
    min_tokens: int,
    max_tokens: int,
    forbidden_words: Tuple[str, ...],
):
    if tree["nodeType"] in ("NP", "NNP", "FRAG"):
        proposal = clean_sentence(tree["word"])
        num_tokens = len(tokenizer.tokenize(proposal))

        if (
            # "." not in proposal
            min_tokens <= num_tokens
            and num_tokens <= max_tokens
            and all(word not in proposal for word in forbidden_words)
        ):
            return proposal
    return None


def retrieve_noun_phrases(
    sentence: str,
    tree: Dict,
    tokenizer: PreTrainedTokenizerBase,
    min_tokens: int,
    max_tokens: int,
    forbidden_words: Tuple[str, ...],
):
    """
    Return a dictionary with noun phrases and the spanning positions
    """
    token = create_token(tree, tokenizer, min_tokens, max_tokens, forbidden_words)
    if token is not None:
        return [token]

    if "children" not in tree:
        return []

    noun_phrases = []

    for children in tree["children"]:
        # if children["nodeType"] not in ("ADJP", "PP"):
        if children["nodeType"] not in ("ADJP",):
            noun_phrases += retrieve_noun_phrases(
                sentence, children, tokenizer, min_tokens, max_tokens, forbidden_words
            )

    return noun_phrases


def is_empty(sentence: str) -> bool:
    return sentence.strip() == ""


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def extracting_noun_phrases(
        sentences: List[Dict[str, Any]], args: Arguments, cuda_device: int
):
    """
    Extract every noun phrases on the given sentences
    """
    # load models
    predictor = Predictor.from_path(
        args.cache_dir / args.parser, cuda_device=cuda_device
    )
    predictor.max_length = args.max_instr_length  # type: ignore
    predictor._json_to_instance = types.MethodType(_json_to_instance, predictor)  # type: ignore
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise ValueError(f"Unexpected type for tokenizer {type(tokenizer)}")

    # extract the noun phrases
    inputs = []
    for stc in sentences:
        if is_empty(stc["sentence"]):
        # if len(stc["instructions"]) == 0:
            stc["noun_phrases"] = []
        else:
            inputs.append(stc)

    total = math.ceil(len(sentences) / args.batch_size)
    for sub in tqdm(batch(inputs, n=args.batch_size), total=total,):
        preds = predictor.predict_batch_json(sub)
        for pred, s in zip(preds, sub):
            s["noun_phrases"] = retrieve_noun_phrases(
                s["sentence"],
                pred["hierplane_tree"]["root"],
                tokenizer,
                args.min_tokens,
                args.max_tokens,
                args.forbidden_words,
            )



def select_best_noun_phrases(
    samples: List[Dict[str, Any]], args: Arguments
):
    """
    Given a bunch of noun phrases, we tried to select the best (or to reject all noun phrases)
    """
    # turn is causing a lot of confusion to the parser
    forbidden_words: Tuple = ("turn",)
    for i, s in enumerate(tqdm(samples)):
        samples[i]["noun_phrases"] = [
            n for n in s["noun_phrases"] if not any(w in n for w in forbidden_words)
        ]

    # we want to prioritize phrases that refer to known objects
    objects_and_rooms: List[str] = load_txt(args.cache_dir / args.categories)
    for i, sample in enumerate(samples):
        if sample["noun_phrases"] == []:
            samples[i]["sentence"] = ""
            continue

        flags = [any(w in n for w in objects_and_rooms) for n in sample["noun_phrases"]]

        if sum(flags) > 0:
            samples[i]["sentence"] = random.choice(
                [n for n, f in zip(sample["noun_phrases"], flags) if f]
            )
        elif sum(flags) == 0:
            samples[i]["sentence"] = random.choice(sample["noun_phrases"])


def clean_category(stc):
    stc = re.sub(";|\?|[0-9]", "", stc)
    stc = re.sub("\((.*)\)", "\1", stc)
    stc = re.sub("  ", " ", stc)
    return stc.strip()


def build_categories(args: Arguments):
    if not (args.cache_dir / args.matterport).is_file():
        download_file(
            "https://github.com/niessner/Matterport/raw/master/metadata/category_mapping.tsv",
            args.cache_dir / args.matterport,
        )
    if not (args.cache_dir / args.places365).is_file():
        download_file(
            "https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt",
            args.cache_dir / args.places365,
        )

    categories = []

    with open(args.cache_dir / args.matterport, newline="") as fid:
        reader = csv.reader(fid, delimiter="\t")
        fieldnames = next(reader)
        for row in reader:
            item = dict(zip(fieldnames, row))
            cat = item["raw_category"].replace("\\", "/").split("/")
            cat = [clean_category(c) for c in cat]
            cat = [c for c in cat if len(c) > 2]
            categories += cat

    with open(args.cache_dir / args.places365) as fid:
        for line in fid.readlines():
            name = line[3:].replace("_", " ")
            name = re.sub(r"\d", "", name)
            name = name.split("/")[0]
            name = name.strip()
            if len(name) > 2:
                categories.append(name)

    # save_txt(set(categories), args.cache_dir / args.categories, ["listing_id", "photo_id", "url", "sentence"])
    save_txt(set(categories), args.cache_dir / args.categories)


def run_extraction(args: Arguments, local_rank: int):
    # fieldnames = ["listing_id", "photo_id", "url", "sentence"]
    fieldnames = ["instr_id", "sentence", "noun_phrases"]

    if not (args.cache_dir / args.categories).is_file():
        build_categories(args)

    if not (args.parser).is_dir():
        args.parser.mkdir(parents=True)
        download_file(
            "https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz",
            args.parser / "parser.tar.gz",
        )
        tf = tarfile.open(args.parser / "parser.tar.gz")
        tf.extractall(args.parser)

    # Load sentences
    start = max(local_rank, 0) + args.start
    # data = load_tsv(args.source, fieldnamesc)
    data = json.load(open(args.source))
    print(start, args.num_splits, len(data), len(data[start:: args.num_splits]))
    data = data[start:: args.num_splits]
    # for sample in data:
    #     sample["sentence"] = clean_sentence(sample["sentence"])

    new_data = []
    for item in data:
        for ind, instr in enumerate(item['instructions']):
            new_data.append({
                'instr_id': str(item['path_id']) + '_' + str(ind),
                'sentence': instr
            })
    print(start, args.num_splits, len(new_data), len(new_data[start:: args.num_splits]))

    extracting_noun_phrases(new_data, args, start % args.num_workers)
    final_data = []
    for item in new_data:
        if len(item['noun_phrases']) > 0:
            final_data.append(item)
    print(len(final_data))

    # select_best_noun_phrases(new_data, args)

    print("Exporting noun phrases to ", args.output)
    output  = args.output.parent / args.output
    save_tsv(final_data, output, fieldnames)


if __name__ == "__main__":
    args = Arguments()

    local_rank = 0
    if local_rank <= 0:
        print(args.to_string(width=80))

    args.cache_dir.mkdir(exist_ok=True, parents=True)


    run_extraction(args, local_rank)
