import torch
from torch import nn
from torchvision.ops import generalized_box_iou
from scipy.optimize import linear_sum_assignment

from utils.bbox import box_cxcywh_to_xyxy

class HungarianMatcher(nn.Module):
    cost_class: float
    cost_bbox: float
    cost_giou: float

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()

        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
    
    def forward(self, outputs: tuple[torch.Tensor, torch.Tensor], targets: list[dict[str, torch.Tensor]]):
        batch_size, num_queries = outputs[0].shape[:2]

        out_prob = outputs[0].flatten(0, 1).softmax(-1)

        out_bbox = outputs[1].flatten(0, 1)

        tgt_ids = torch.cat([v['labels'] for v in targets])
        tgt_bbox = torch.cat([v['boxes'] for v in targets])

        cost_class = -out_prob[:, tgt_ids]

        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(batch_size, num_queries, -1)

        sizes = [len(v['boxes']) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
            ]
        
        return indices
