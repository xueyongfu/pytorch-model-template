from sklearn.metrics import accuracy_score, f1_score


def compute_cls_metrics(lables, logits, task_name):
    return {task_name + '/acc': accuracy_score(lables, logits),
            task_name + '/f1': f1_score(lables, logits, average='macro')}


def compute_ner_metrics(labels, logits, lens, task_name):
    from seqeval.metrics import f1_score
    y_pred = []
    y_true = []
    for label, logit, length in zip(labels, logits, lens):
        y_pred += logit[:length][1:-1]
        y_true += label[:length][1:-1]
    f1 = f1_score(y_true, y_pred)
    return {task_name + '/f1': f1}


class Metrics():
    def __init__(self, task_name):
        self.task_name = task_name
        self.values = dict()
        self.count = 0

    def update(self, labels, logits, type, lens):
        self.count += 1
        if type == 'classification':
            metrics = compute_cls_metrics(labels, logits, self.task_name)
        else:
            metrics = compute_ner_metrics(labels, logits, lens, self.task_name)
        for k, v in metrics.items():
            self.values[k] = self.values[k] + v if self.values.get(k) else v

    @property
    def metrics(self):
        for k, v in self.values.items():
            self.values[k] = self.values[k] / self.count
        return self.values
