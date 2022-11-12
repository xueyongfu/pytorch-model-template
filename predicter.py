import torch
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from utils.logger import logger
from data_processor.datasets import load_and_cache_examples


def predict(args, task_params, model, tokenizer, mode, prefix):
    eval_dataset = load_and_cache_examples(args, task_params, tokenizer, mode)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset,
                                 sampler=eval_sampler,
                                 batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running test %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()

    eval_steps = 0
    all_preds = {name: list() for name in task_params.keys()}
    all_lens = list()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet"] else None
                )  # XLM and RoBERTa don"t use segment_ids

            if args.single_task == 0:
                labels_dict = dict(zip(task_params.keys(), batch[3:]))
                logits_dict, eval_loss = model(inputs=inputs, labels=labels_dict, mode=mode)
            else:
                labels_dict = dict(zip(task_params.keys(), batch[3:-1]))
                logits_dict, eval_loss = model(inputs=inputs, labels=labels_dict, lens=batch[-1], mode=mode)

            for key in task_params.keys():
                lens = batch[-1].detach().cpu().numpy().tolist()
                # 存放所有preds
                all_preds[key].extend(logits_dict[key])
                # 存放所有lens
                all_lens.extend(lens)

            eval_steps += 1

    return None, all_preds, all_lens
