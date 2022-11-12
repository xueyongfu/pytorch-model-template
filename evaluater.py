import torch
from tqdm import tqdm

from hyperparameter import args, task_params
from utils.logger import logger


def evaluate(eval_dataloader, model, prefix):
    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()

    eval_total_loss = 0
    eval_steps = 0
    y_true = None
    y_pred = None

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = {n: t.to(args.device) for n, t in batch.items()}
        with torch.no_grad():

            loss, outputs = model(batch, mode='eval')
            labels = outputs['labels']
            logits = outputs['logits']
            pred_labels = torch.argmax(logits, dim=-1)

            y_true = labels if y_true is None else torch.cat([y_true, labels])
            y_pred = pred_labels if y_pred is None else torch.cat([y_pred, pred_labels])

            if args.n_gpu > 1:
                eval_loss = eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
            eval_total_loss += eval_loss.item()
            eval_steps += 1

    metrics = {}
    metrics['f1'] = 0

    return loss, metrics
