import torch
import os
import torchvision
from PIL import Image


class BoxesDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.imgs = os.listdir(os.path.join(self.root, "images"))
        self.labels = os.listdir(os.path.join(self.root, "labels"))

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
