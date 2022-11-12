import os
import torch
from torch.utils.data import Dataset

from data_processor.data_reader import read_examples_from_file
from utils.logger import logger


def collate_func(batch):
    # convert to tensor, batch

    return {}


def load_and_cache_examples(args, task_params, tokenizer, mode):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}".format(
            mode, list(filter(None, args.model_name_or_path.split("/"))).pop(), str(args.max_seq_length)
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        dataset = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        data_list = read_examples_from_file(args, task_params, tokenizer, mode)
        dataset = DatasetList(data_list)

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save((dataset), cached_features_file)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    return dataset


class DatasetList(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, idx):
        return self.data_list[idx]

    def __len__(self):
        return len(self.data_list)
