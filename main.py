import os

os.environ['CUDA_VISIBLE_DEVICES'] = '11'

import pandas as pd
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel

from data_processor.datasets import load_and_cache_examples
from trainer import train
from hyperparameter import args, task_params
from utils.logger import logger
from utils.utils import set_seed


def main():
    set_seed(args)

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1  # DDP模式下，一个进程一个GPU
    args.device = device

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModel.from_pretrained(args.model_name_or_path, config=config, args=args, task_params=task_params)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        eval_dataset = load_and_cache_examples(args, task_params, tokenizer, mode='eval')
        train_dataset = load_and_cache_examples(args, task_params, tokenizer, mode="train")
        global_step = train(train_dataset, eval_dataset, model, tokenizer)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        output_dir = os.path.join(args.output_dir, "steps-%d" % global_step)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", output_dir)

        # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        # logger.info("Saving optimizer and scheduler states to %s", output_dir)


if __name__ == "__main__":
    main()
