from typing import Optional
import torch
import random
import matplotlib.pyplot as plt

from .bbox import bbox_to_rect


def show_sample(
        xb: torch.Tensor,
        yb: list[dict[str, torch.Tensor]],
        classes: list[str],
        idx: Optional[int] = None
):
    """
    Arguments:
        xb: Tensor, shape `[batch_size, num_channels, height, width]`
        yb: Tuple[labels, bboxes]

            -labels: Tensor, shape `[batch_size, num_objects]`
            -bboxes: Tensor, shape `[batch_size, num_objects, 4]`
    """

    N, _, H, W = xb.shape
    sample_idx = random.randint(0, N-1) if idx is None else idx

    image = xb[sample_idx]

    target = yb[sample_idx]
    labels = target['labels']
    bboxes = target['bboxes']

    rects = bbox_to_rect(labels, bboxes, classes, size=(W, H))

    fig, ax = plt.subplots()

    ax.imshow(image.permute(1, 2, 0))

    for text, rect in rects:
        ax.add_patch(rect)
        ax.text(**text)
