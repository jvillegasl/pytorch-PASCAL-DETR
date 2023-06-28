import torch
from torch.utils.data.dataloader import default_collate
from torchvision import datasets, transforms

from base import BaseDataLoader
from data_loader.collate_fn import Collator


class PascalDataLoader(BaseDataLoader):
    classes = [
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
    max_num_objects = 100

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224))
        ])

        image_set = 'trainval' if training else 'test'

        self.data_dir = data_dir
        self.dataset = datasets.VOCDetection(
            self.data_dir, year='2007', image_set=image_set, download=True, transform=trsfm)

        super().__init__(
            self.dataset,
            batch_size,
            shuffle,
            validation_split,
            num_workers,
            collate_fn=Collator(self.classes, self.max_num_objects)
        )
