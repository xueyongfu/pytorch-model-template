import torch.nn as nn
import torch
import torch.nn.functional as F


def calculate_auto_weight_total_loss(loss_list):
    def weight_loss(loss, weight):
        return torch.exp(-weight) * loss + weight - 1

    addressEntity_loss, influence_loss, requestion_loss, houseType_loss, num_loss = loss_list
    total_loss = weight_loss(addressEntity_loss, addressEntityWeight) + \
                 weight_loss(influence_loss, influenceWeight) + weight_loss(requestion_loss,
                                                                            requestionWeight) + \
                 weight_loss(houseType_loss, houseTypeWeight) + weight_loss(num_loss, numWeight)

    return total_loss


def calculate_custom_weight_total_loss(loss_dict, task_params):
    total_loss = 0
    for key, value in loss_dict.items():
        total_loss += value * task_params[key]['weight']
    return total_loss


class CeCriterion():
    def __init__(self, alpha=1.0, name='Cross Entropy Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1):
        """weight: sample weight
        """
        if weight:
            loss = torch.mean(F.cross_entropy(input, target, reduce=False, ignore_index=ignore_index) * weight)
        else:
            loss = F.cross_entropy(input, target, ignore_index=ignore_index)
        loss = loss * self.alpha
        return loss


LOSS_MAP = {'CrossEntropy':CeCriterion}