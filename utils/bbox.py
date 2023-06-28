from typing import Literal
import matplotlib.patches as patches
import torch


def bbox_xyxy_to_xywh(bboxes: torch.Tensor):
    """
    Arguments:
        bbox: Tensor, shape `[..., 4]`

    Returns:
        Tensor, shape `[..., 4]`
    """

    xmin, ymin, xmax, ymax = bboxes.unbind(-1)

    width = xmax - xmin
    height = ymax - ymin

    xcenter = (xmax + xmin) / 2
    ycenter = (ymax + ymin) / 2

    return torch.stack([xcenter, ycenter, width, height], dim=-1)


def bbox_xywh_to_xyxy(bboxes: torch.Tensor):
    """
    Arguments:
        bbox: Tensor, shape `[..., 4]`

    Returns:
        Tensor, shape `[..., 4]`
    """

    xcenter, ycenter, width, height = bboxes.unbind(-1)

    xmin = xcenter - width / 2
    xmax = xcenter + width / 2

    ymin = ycenter - height / 2
    ymax = ycenter + height / 2

    return torch.stack([xmin, ymin, xmax, ymax], dim=-1)


def box_cxcywh_to_xyxy(x: torch.Tensor):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: torch.Tensor):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def bbox_to_rect(
        bboxes: torch.Tensor,
        labels: torch.Tensor,
        classes: list[str],
        size: tuple[int, int],
):
    """
    Arguments:
        bboxes: Tensor, shape `[num_objects, 4]`
        labels: Tensor, shape `[num_objects, 1]`
        classes: List[str]
        size: Tuple[int, int], format `[width, height]`

    Returns:
        rectangles: list[tuple[dict, Rectangle]]
    """

    objects = torch.cat([bboxes, labels], dim=-1)

    W, H = size

    rectangles: list[tuple[dict, patches.Rectangle]] = []

    for obj in objects:
        xmin, ymin, xmax, ymax, label = obj.tolist()
        label = int(label)

        if label >= len(classes):
            continue
        class_name = classes[label]

        w, h = xmax - xmin, ymax - ymin

        x, y = xmin*W, ymin*H
        w, h = w*W, h*H

        rect = patches.Rectangle(
            [x, y], w, h, linewidth=1, edgecolor='r', facecolor='none')

        text_args = {
            'x': x,
            'y': y,
            's': class_name,
            'fontsize': 10,
            'bbox': dict(facecolor='r', alpha=0.5)
        }

        rectangles.append((text_args, rect))

    return rectangles
