import torch
from torch.utils.data.dataloader import default_collate


def collate_fn(batch: list, classes: list[str], max_num_objects: int):
    batch = list(map(
        lambda x: [
            x[0],
            annotation_to_tensor(
                x[1]['annotation'],
                classes,
                max_num_objects
            )
        ],
        batch
    ))

    return default_collate(batch)


def annotation_to_tensor(annotation: dict, classes: list[str],  max_num_objects: int):
    size = annotation['size']
    W, H = int(size['width']), int(size['height'])

    objects = annotation['object']

    objects = list(map(lambda x: object_to_tensor(x, classes, W, H), objects))
    objects = objects[:max_num_objects]

    null_object = [
        torch.tensor([len(classes)]),
        torch.zeros(4)
    ]

    while (len(objects) < max_num_objects):
        objects.append(null_object)

    out = [
        torch.cat([t[0].unsqueeze(0) for t in objects]),
        torch.cat([t[1].unsqueeze(0) for t in objects]),
    ]

    return out


def object_to_tensor(object: dict, classes: list[str], height: int, width: int):
    class_name = object['name']
    bbox = object['bndbox']

    class_index = classes.index(class_name)
    class_index = torch.tensor([class_index])

    xmin = int(bbox['xmin'])
    ymin = int(bbox['ymin'])
    xmax = int(bbox['xmax'])
    ymax = int(bbox['ymax'])

    bbox = [xmin/width, ymin/height,
            xmax/width, ymax/height]
    bbox = torch.tensor(bbox)

    return [class_index, bbox]
