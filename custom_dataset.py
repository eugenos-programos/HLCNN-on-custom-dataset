import torch
import os
import torchvision
from PIL import Image
import numpy as np
from skimage import transform as sktransform



class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.imgs = os.listdir(os.path.join(self.root, "images"))
        self.labels = os.listdir(os.path.join(self.root, "labels"))

    def preprocess(self, img, min_size=720, max_size=1280):
        H, W, C = img.shape
        scale1 = min_size / min(H, W)
        scale2 = max_size / max(H, W)
        scale = min(scale1, scale2)
        img = img / 255.
        img = sktransform.resize(
            img, (int(H * scale), int(W * scale), C), mode='reflect', anti_aliasing=True)
        img = np.asarray(img, dtype=np.float32)
        return img

    def resize_bbox(self, bbox, in_size, out_size):
        bbox = bbox.copy()
        x_scale = float(out_size[0]) / in_size[0]
        y_scale = float(out_size[1]) / in_size[1]
        bbox[:, 0] = np.round(y_scale * bbox[:, 0])
        bbox[:, 2] = np.round(y_scale * bbox[:, 2])
        bbox[:, 1] = np.round(x_scale * bbox[:, 1])
        bbox[:, 3] = np.round(x_scale * bbox[:, 3])
        return bbox

    def read_gt_bbox(self, annoFile):
        gt_boxes = []
        for line in annoFile:
            bbox = line.split()
            gt_boxes.append([int(bbox[0])+1, int(bbox[1])+1, int(bbox[2]) -
                            int(bbox[0])+1, int(bbox[3])-int(bbox[1]), int(bbox[4])])
        return gt_boxes

    def __transform_boxes_coords__(self, boxes_coords, img_size=640):
        center_x, center_y, width, height = boxes_coords
        x_1 = center_x - width / 2
        x_2 = center_x + width / 2
        y_1 = center_y - height / 2
        y_2 = center_y + height / 2
        if x_1 >= x_2 or y_1 >= y_2:
            raise ValueError("Incorrect coords")
        boxes_coords = list(
            map(lambda coordinate: coordinate * img_size, [x_1, y_1, x_2, y_2]))
        return boxes_coords

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        label_path = os.path.join(self.root, "labels", self.labels[idx])
        img = Image.open(img_path).convert("RGB")
        with open(label_path) as lfile:
            boxes_coords = lfile.readlines()
        boxes_coords = [bcoord_line[:-1].split()[1:]
                        for bcoord_line in boxes_coords]
        boxes_coords = [list(map(float, box_coords))
                        for box_coords in boxes_coords]
        boxes_coords = [self.__transform_boxes_coords__(
            box_coords) for box_coords in boxes_coords if box_coords[3] != 0 and boxes_coords[2] != 0]
        boxes = torch.as_tensor(boxes_coords, dtype=torch.float32)
        labels = torch.zeros((boxes.shape[0]), dtype=torch.int64)
        image_id = torch.tensor([idx])
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id
        }

        return torchvision.transforms.functional.pil_to_tensor(img) / 256., target

    def __len__(self):
        return len(self.imgs)
