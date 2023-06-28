import torch
from torch.utils.data.dataloader import default_collate


class PascalCollator(object):
    def __init__(self, classes: list[str]):
        self.classes = classes

    def __call__(self, batch):
        return collate_fn(batch, self.classes)


def collate_fn(batch: list, classes: list[str]):
    batch = list(map(
        lambda x: [
            x[0],
            annotation_to_batch_item(x[1], classes)
        ],
        batch
    ))

    inputs = default_collate([t[0] for t in batch])
    targets = [t[1] for t in batch]

    return inputs, targets


def annotation_to_batch_item(annotation_dict: dict, classes: list[str]):
    annotation = annotation_dict['annotation']

    size = annotation['size']
    W, H = int(size['width']), int(size['height'])

    objects = annotation['object']
    objects = list(map(lambda x: object_to_tensor(x, classes, H, W), objects))

    labels_ids = torch.cat([t[0].unsqueeze(0) for t in objects])
    bboxes = torch.cat([t[1].unsqueeze(0) for t in objects])

    return {'labels': labels_ids, 'bboxes': bboxes}


def object_to_tensor(object: dict, classes: list[str], height: int, width: int):
    class_name = object['name']
    bbox = object['bndbox']

    class_index = classes.index(class_name)
    class_index = torch.as_tensor(class_index, dtype=torch.int64)

    xmin = int(bbox['xmin'])
    ymin = int(bbox['ymin'])
    xmax = int(bbox['xmax'])
    ymax = int(bbox['ymax'])

    bbox = [xmin/width, ymin/height,
            xmax/width, ymax/height]
    bbox = torch.tensor(bbox)

    return (class_index, bbox)
