import numpy as np
import torch.nn as nn

from linear import Linear
from crf import CRF
from loss import calculate_custom_weight_total_loss
from transformers import BertModel, BertPreTrainedModel


class MultiTaskModel(BertPreTrainedModel):
    def __init__(self, config, args, task_params):
        super().__init__(config)
        self.task_params = task_params
        self.bert = BertModel(config)
        self.dropout = nn.Dropout()
        self.loss_fct = nn.CrossEntropyLoss()

        self.modelist = nn.ModuleList()
        self.crf_layer = nn.ModuleList()

        for key in task_params.keys():
            if self.task_params[key]['type'] == 'classification':
                self.modelist.append(
                    nn.Linear(config.hidden_size, task_params[key]['label_nums']))
                self.crf_layer.append(nn.Module())
            else:
                self.modelist.append(
                    nn.Linear(config.hidden_size, task_params[key]['label_nums']))
                self.crf_layer.append(CRF(tagset_size=task_params[key]['label_nums'],
                                          tag_dictionary=task_params[key]['label2id'],
                                          device=args.device,
                                          is_bert=True))

    def forward(self, inputs, labels, lens=None, mode='dev'):
        sequence_output, hidden_output = self.bert(**inputs)
        sequence_output, hidden_output = self.dropout(sequence_output), self.dropout(hidden_output)

        loss_dict = dict()
        logits_dict = dict()
        for key, model, crf in zip(self.task_params.keys(), self.modelist, self.crf_layer):
            if self.task_params[key]['type'] == 'classification':
                logits = model(hidden_output)
                if mode == 'test':
                    logits_dict[key] = np.argmax(logits.detach().cpu().numpy(), axis=-1)
                else:
                    loss = self.loss_fct(logits, labels[key])
                    loss_dict[key] = loss
                    if mode == 'dev':
                        logits_dict[key] = np.argmax(logits.detach().cpu().numpy(), axis=-1)
            else:
                logits = model(sequence_output)
                if mode == 'test':
                    logits_dict[key] = crf._obtain_labels(logits, self.task_params[key]['id2label'], lens)[0]
                else:
                    loss = crf.calculate_loss(logits, labels[key], lens)
                    loss_dict[key] = loss
                    if mode == 'dev':
                        logits_dict[key] = crf._obtain_labels(logits, self.task_params[key]['id2label'], lens)[0]
        if mode == 'test':
            return logits_dict, None
        total_loss = calculate_custom_weight_total_loss(loss_dict, self.task_params)
        return logits_dict, total_loss
