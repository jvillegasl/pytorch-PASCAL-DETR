import torch
import random

import inject_modules
from data_loader.collate_fn import collate_fn, annotation_to_tensor, object_to_tensor
from mock_data import CLASSES, mock_object, mock_dataset, mock_annotation, mock_image


class TestCollateFn:
    dataset: list[list]
    batch_size = 64

    chw = (3, 224, 224)

    classes = CLASSES
    max_num_objects = 100

    @classmethod
    def setup_class(cls):
        cls.dataset = mock_dataset(
            cls.batch_size, cls.chw, cls.max_num_objects)

    def test_object_to_tensor(self):
        H, W = random.randint(0, 1000), random.randint(0, 1000)

        obj = mock_object(H, W)
        tensor = object_to_tensor(obj, self.classes, H, W)

        assert isinstance(tensor, list)
        assert len(tensor) == 2

        class_index, bbox = tensor
        assert isinstance(class_index, torch.Tensor)
        assert isinstance(bbox, torch.Tensor)

        assert list(class_index.shape) == [1]
        assert list(bbox.shape) == [4]
        assert torch.all(torch.logical_and(
            bbox >= 0,
            bbox <= 1
        ))

    def test_annotation_to_tensor(self):
        C, H, W = 3, random.randint(0, 1000), random.randint(0, 1000)

        image = mock_image(C, H, W)
        annotation = mock_annotation(image, self.max_num_objects)['annotation']

        out = annotation_to_tensor(
            annotation, self.classes, self.max_num_objects)

        assert isinstance(out, list)
        assert len(out) == 2
        assert isinstance(out[0], torch.Tensor)
        assert isinstance(out[1], torch.Tensor)

        classes = out[0]
        bboxes = out[1]

        assert list(classes.shape) == [self.max_num_objects, 1]
        assert list(bboxes.shape) == [self.max_num_objects, 4]
        assert torch.all(torch.logical_and(
            bboxes >= 0,
            bboxes <= 1
        ))

    def test_collate_fn(self):
        batch = collate_fn(self.dataset, self.classes)

        xb, yb = batch

        assert isinstance(xb, torch.Tensor)

        xb_shape = xb.shape
        assert list(xb_shape) == [self.batch_size, *self.chw]

        assert isinstance(yb, list)
        assert len(yb) == 2
        assert isinstance(yb[0], torch.Tensor) and isinstance(
            yb[1], torch.Tensor)

        yb_classes, yb_bboxes = yb[0], yb[1]

        assert yb_classes.size(0) == self.batch_size and yb_classes.size(2) == 1
        assert yb_bboxes.size(0) == self.batch_size and yb_bboxes.size(2)  == 4
        assert torch.all(torch.logical_and(
            yb_bboxes >= 0,
            yb_bboxes <= 1
        ))
