import torch
from torch import nn

class HungarianMatcher(nn.Module):

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1):
        pass