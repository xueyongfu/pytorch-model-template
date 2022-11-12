import os
import numpy as np
import random
import pandas as pd
import yaml
import torch


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)





def get_entity_bio(seq):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def ner_entity(tags, words, all_lens):
    labels = []
    for tag, word, length in zip(tags, words, all_lens):
        trunc_word = word[:length - 2]
        trunc_tags = tag[:length][1:-1]
        ents = get_entity_bio(trunc_tags)
        tmp = []
        for ent in ents:
            tmp.append(''.join(trunc_word[ent[1]: ent[2] + 1]))
        tmp = [ent for ent in set(tmp) if len(ent) > 1]
        if tmp == []:
            labels.append('')
        elif len(tmp) == 1:
            labels.append(tmp[0])
        else:
            labels.append(':'.join(tmp))
    return labels


def convert_ids_to_label(ids, task_name, task_params):
    labels = []
    for row in ids:
        if task_params[task_name]['type'] == 'classification':
            labels.append(task_params[task_name]['id2label'][row])
        else:
            labels.append([task_params[task_name]['id2label'][l] for l in row])
    return labels
