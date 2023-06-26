import torch
from torch import nn
import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)


def bipartite_loss(output, target):
    """
    Arguments:
        output: Tuple(logits, bboxes)

            -logits: Tensor, shape `[batch_size, max_num_objects, num_classes]`
            -bboxes: Tensor, shape `[batch_size, max_num_objects, 4]`

        target: Tuple(logits, bboxes)

            -logits: Tensor, shape `[batch_size, max_num_objects]`
            -bboxes: Tensor, shape `[batch_size, max_num_objects]`

    Returns:
        loss: Tensor, shape `[1]`
    """


class Criterion(nn.Module):

    def __init__(self, num_classes: int = 20, eos_coef: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.eos_coef = eos_coef

        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, output, target):
        """
        Arguments:
            output: Tensor, shape `[batch_size, max_num_objects, num_classes]`

            target: Tensor, shape `[batch_size, max_num_objects`

        Returns:
            loss, Tensor, shape `[1]`
        """

        
