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

    def loss_labels(
            self,
            outputs: dict[str, torch.Tensor],
            targets: list[dict[str, torch.Tensor]],
            indices: list[tuple[torch.Tensor, torch.Tensor]],
    ):
        """Classification loss (NLL)
        Arguments:
            outputs: Dict containing:
               - "pred_logits": Tensor, shape `[batch_size, num_queries, num_classes]`
               - "pred_bboxes": Tensor, shape `[batch_size, num_queries, 4]`

            targets: List[dict], `len(targets) == batch_size`, each dict contains:
               - "labels": Tensor, shape `[num_objects_i]`
               - "bboxes": Tensor, shape `[num_objects_i, 4]`

            indices: List[Tuple[Tensor, Tensor]] where `len(indices) == batch_size` and:
               - indices[i][0]: Indices of the selected predictions (in order)
               - indices[i][1]: Indices of the corresponding selected targets (in order)

            For each batch element, it holds:
                indices[i][0].shape == indices[i][1].shape = min(num_queries, num_objects_i)
        Returns:
            losses: Dict containing:
               - "loss_ce": Tensor, shape `[]`
        """

        pred_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        # Tuple[Tensor[total_num_objects], Tensor[total_num_objects]]
        # Tuple[batch_coords, pred_coords]

        zipped_tgt_idx = zip(targets, indices)
        # Zip[Tuple[target[i], indices[i]]]

        target_classes_o = torch.cat([
            t['labels'][J]
            for t, (_, J) in zipped_tgt_idx
        ])
        # ordered target classes
        # [total_num_objects]

        target_classes = torch.full(
            pred_logits.shape[:2], self.num_classes, dtype=torch.int64)
        # filled with no-object class
        # [batch_size, num_queries]

        target_classes[idx] = target_classes_o
        # idx indicates the coords (batch, query)
        # to be replaced with the corresponding target class

        loss_ce = F.cross_entropy(
            pred_logits.transpose(1, 2),
            target_classes,
            self.empty_weight
        )

        losses = {'loss_ce': loss_ce}

        return losses

    def loss_cardinality(
            self,
            outputs: dict[str, torch.Tensor],
            targets: list[dict[str, torch.Tensor]]
    ):
        """
        Arguments:
            outputs: Dict containing:
               - "pred_logits": Tensor, shape `[batch_size, num_queries, num_classes]`
               - "pred_bboxes": Tensor, shape `[batch_size, num_queries, 4]`

            targets: List[dict], `len(targets) == batch_size`, each dict contains:
               - "labels": Tensor, shape `[num_objects_i]`
               - "bboxes": Tensor, shape `[num_objects_i, 4]`
        """

        pred_logits = outputs['pred_logits']

        tgt_lengths = torch.as_tensor([len(v['labels']) for v in targets])

        # Count the number of predictions that are NOT 'no-object' (which is the last class)
        pred_labels = pred_logits.argmax(-1)
        # [batch_size, num_queries]
        no_object_label = pred_logits.size(-1) - 1

        card_pred = (pred_labels != no_object_label).sum(1)
        # [batch_size]
        # count of NOT 'no-object' predictions

        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        # []

        losses = {'cardinality_error': card_err}

        return losses

    def _get_src_permutation_idx(self, indices: list[tuple[torch.Tensor, torch.Tensor]]):
        """
        Arguments:
            indices: List[Tuple[Tensor, Tensor]] where `len(indices) == batch_size` and:
               - indices[i][0]: Indices of the selected predictions (in order)
               - indices[i][1]: Indices of the corresponding selected targets (in order)

            For each batch element, it holds:
                indices[i][0].shape == indices[i][1].shape = min(num_queries, num_objects_i)
        """

        batch_idx = torch.cat([
            torch.full_like(src, i)
            for i, (src, _) in enumerate(indices)
        ])
        # batch coordinates
        # [total_num_objects]

        src_idx = torch.cat([src for (src, _) in indices])
        # prediction coordinates
        # [total_num_objects]

        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices: list[tuple[torch.Tensor, torch.Tensor]]):
        batch_idx = torch.cat([
            torch.full_like(tgt, i)
            for i, (_, tgt) in enumerate(indices)
        ])

        tgt_idx = torch.cat([tgt for (_, tgt) in indices])

        return batch_idx, tgt_idx
