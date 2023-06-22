import random

import torch

CLASSES = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor'
]


def mock_class_name():
    return random.choice(CLASSES)


def mock_bbox(height: int, width: int):
    assert width > 0, 'width must be a positive integer'
    assert height > 0, 'height must be a positive integer'

    xmin, xmax = generate_min_max(width)
    ymin, ymax = generate_min_max(height)

    return xmin, ymin, xmax, ymax


def mock_image(channels: int, height: int, width: int):
    return torch.rand(channels, height, width)

def mock_object(height: int, width: int):
    xmin, ymin, xmax, ymax = mock_bbox(height, width)

    obj = {
        'name': mock_class_name(),
        'bndbox': {
            'xmin': str(xmin),
            'ymin': str(ymin),
            'xmax': str(xmax),
            'ymax': str(ymax),
        }
    }

    return obj


def mock_annotation(image: torch.Tensor, max_num_objects: int):
    assert max_num_objects > 0, 'max_num_objects must be a positive integer'

    C, H, W = image.shape

    num_objects = random.randint(1, max_num_objects)
    objects = [mock_object(H, W) for _ in range(num_objects)]

    annotation = {
        'size': {
            'width': str(W),
            'height': str(H),
            'depth': str(C),
        },
        'object': objects
    }

    annotation = {
        'annotation': annotation
    }

    return annotation


def mock_dataset(n_samples: int, chw: tuple[int, int, int], max_num_objects: int):
    assert n_samples > 0, 'n_samples must be a positive integer'
    assert max_num_objects > 0, 'max_num_objects must be a positive integer'

    C, H, W = chw

    dataset = []
    for _ in range(n_samples):
        image = mock_image(C, H, W)
        annotation = mock_annotation(image, max_num_objects)
        data = [image, annotation]

        dataset.append(data)

    return dataset


def generate_min_max(x: int):
    min = random.randint(0, x)
    max = random.randint(min, x)
    return min, max

