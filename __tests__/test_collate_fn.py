import torch
import random

import inject_modules
from data_loader.collate_fn import collate_fn, annotation_to_batch_item
from mock_data import CLASSES, mock_dataset, mock_annotation, mock_image
from utils.check_exception import check_exception


class TestCollateFn:
    def test_annotation_to_batch_item(self):
        C = 3
        H = random.randint(100, 1000)
        W = random.randint(100, 1000)
        max_num_objects = random.randint(10, 100)
        classes = CLASSES

        num_objects = random.randint(1, max_num_objects)

        image = mock_image(C, H, W)
        annotation = mock_annotation(image, num_objects)

        x = annotation_to_batch_item(annotation, classes)

        x = self.assert_annotation_batch_item(x)

        labels_ids = x['labels']
        bboxes = x['bboxes']

        assert labels_ids.shape == torch.Size([num_objects])
        assert bboxes.shape == torch.Size([num_objects, 4])

    def test_collate_fn(self):
        batch_size = random.randint(1, 100)
        C = 3
        H = random.randint(100, 1000)
        W = random.randint(100, 1000)
        max_num_objects = random.randint(10, 100)
        classes = CLASSES

        dataset_batch = mock_dataset(batch_size, (C, H, W), max_num_objects)

        batch = collate_fn(dataset_batch, classes)

        assert isinstance(batch, tuple)
        assert len(batch) == 2

        xb, yb = batch

        assert isinstance(xb, torch.Tensor)
        assert xb.shape == torch.Size((batch_size, C, H, W))

        assert isinstance(yb, list)
        assert len(yb) == batch_size
        assert all(
            [
                all([
                    isinstance(y, dict),
                    'labels' in y and 'bboxes' in y,
                    not check_exception(
                        lambda: self.assert_annotation_batch_item(y),
                        AssertionError
                    )
                ])
                for y in yb
            ]
        )

    def assert_annotation_batch_item(self, x):
        assert isinstance(x, dict)
        assert 'labels' in x and 'bboxes' in x

        labels_ids = x['labels']
        bboxes = x['bboxes']

        assert isinstance(labels_ids, torch.Tensor)
        assert len(labels_ids.shape) == 1
        assert labels_ids.dtype == torch.int64

        assert isinstance(bboxes, torch.Tensor)
        assert len(bboxes.shape) == 2
        assert bboxes.size(0) == labels_ids.size(0)
        assert bboxes.size(1) == 4
        assert torch.all(torch.logical_and(
            bboxes >= 0,
            bboxes <= 1
        ))

        return {'labels': labels_ids, 'bboxes': bboxes}
