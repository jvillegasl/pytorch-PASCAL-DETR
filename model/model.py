import torch
import torch.nn as nn
from torchvision.models import resnet50

from base import BaseModel


class DETR(BaseModel):
    hidden_dim: int = 256
    nheads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    max_num_objects: int = 100

    def __init__(self, num_classes=91):
        super().__init__()

        self.backbone = nn.Sequential(
            *list(resnet50(pretrained=True).children())[:-2])

        self.conv = nn.Conv2d(2048, self.hidden_dim, 1)

        self.transformer = nn.Transformer(
            self.hidden_dim,
            self.nheads,
            self.num_encoder_layers,
            self.num_decoder_layers
        )

        self.linear_class = nn.Linear(self.hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(self.hidden_dim, 4)

        self.query_pos = nn.Parameter(torch.rand(
            self.max_num_objects, self.hidden_dim))
        self.row_embed = nn.Parameter(torch.rand(50, self.hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, self.hidden_dim // 2))

    def forward(self, inputs: torch.Tensor):
        """
        Arguments:
            inputs: Tensor, shape `[batch_size, channels, height, width]`

        Returns:
            outputs: Dict
               - "pred_logits": Tensor, shape `[batch_size, num_queries, num_classes + 1]`
               - "pred_bboxes": Tensor, shape `[batch_size, num_queries, 4]`
        """
        x = self.backbone(inputs)
        h = self.conv(x)

        N, _, H, W = h.shape

        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        pos = pos.repeat(1, N, 1)

        h = self.transformer(
            pos + h.flatten(2).permute(2, 0, 1),
            self.query_pos.unsqueeze(1).repeat(1, N, 1)
        ).permute(1, 0, 2)

        outputs = {
            'pred_logits': self.linear_class(h),
            'pred_bboxes': self.linear_bbox(h).sigmoid()
        }

        return outputs
