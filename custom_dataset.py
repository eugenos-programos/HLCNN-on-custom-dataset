import torch
import os
import torchvision
from PIL import Image
import numpy as np
from skimage import transform as sktransform
from data_aug import *


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
    
    def read_gt_bbox(self, annoFile):
        gt_boxes = []
        for line in annoFile:
            bbox = list(map(float, line.split()))
            gt_boxes.append([int(bbox[0])+1, int(bbox[1])+1, int(bbox[2]) -
                            int(bbox[0])+1, int(bbox[3])-int(bbox[1]), int(bbox[4])])
        return gt_boxes

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        label_path = os.path.join(self.root, "labels", self.labels[idx])
        import cv2
        img = np.array(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))

        H, W, _ = img.shape

        img = self.preprocess(img)
        o_H, o_W, _ = img.shape
        dSR = 1
        GAM = np.zeros((1, int(o_H / dSR), int(o_W / dSR)))

        annoFile = open(label_path)
        gt_bbox = np.asarray(self.read_gt_bbox(annoFile))

        numCar = 0
        if gt_bbox.shape[0] > 0:
            gt_boxes = np.asarray(self.resize_bbox(gt_bbox, (H, W), (o_H, o_W)), dtype=np.float)

            gt_boxes[:, 2] = gt_boxes[:, 0] + gt_boxes[:, 2]
            gt_boxes[:, 3] = gt_boxes[:, 1] + gt_boxes[:, 3]


            import random
            if random.random() > 1:
                    transforms_aug = Sequence([RandomRotate(45)])
                    img, gt_boxes = transforms_aug(img, gt_boxes[:,:4])
            gt_boxes = gt_boxes / dSR
            gt_boxes[:, 0::2] = np.clip(gt_boxes[:, 0::2], 0, int(o_W / dSR))
            gt_boxes[:, 1::2] = np.clip(gt_boxes[:, 1::2], 0, int(o_H / dSR))

            gt_boxes[:, 2] = abs(gt_boxes[:, 2] - gt_boxes[:, 0])
            gt_boxes[:, 3] = abs(gt_boxes[:, 3] - gt_boxes[:, 1])


            gt_boxes = np.delete(gt_boxes, np.where(gt_boxes[:, 3]==0),0)
            gt_boxes = np.delete(gt_boxes, np.where(gt_boxes[:, 2]==0),0)

            numCar = gt_boxes.shape[0]
            # prepare GAM image

            for bbox in gt_boxes:

                bbox = np.asarray(bbox, dtype=np.int)

                dhsizeh = int(bbox[3] / 2)
                dhsizew = int(bbox[2] / 2)

                if dhsizeh % 2 == 0:
                    dhsizeh = dhsizeh + 1

                if dhsizew % 2 == 0:
                    dhsizew = dhsizew + 1

                sigma = np.sqrt(dhsizew * dhsizeh) / (1.96*1.5)
                h_gauss = np.array(twoD_Gaussian(dhsizew, dhsizeh, sigma, math.ceil(dhsizew / 4), math.ceil(dhsizeh / 4)))
                h_gauss = h_gauss / np.max(h_gauss)

                cmin = bbox[1]
                rmin = bbox[0]
                cmax = bbox[1] + int(2*dhsizeh)+1
                rmax = bbox[0] + int(2*dhsizew)+1

                if cmax > int(o_H / dSR):
                    cmax = int(o_H / dSR)

                if rmax > int(o_W / dSR):
                    rmax = int(o_W / dSR)
                GAM[0, cmin:cmax, rmin:rmax] = GAM[0, cmin:cmax, rmin:rmax] + h_gauss[0:cmax-cmin, 0:rmax-rmin]

        downsampler = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(), torchvision.transforms.Resize((int(o_H / 8), int(o_W / 8)), interpolation=Image.LANCZOS)])
        #
        GAM = downsampler(torch.Tensor(GAM))
        GAM = np.array(GAM)
        GAM = (GAM / GAM.max()) * 1
        # plt.imshow(img)
        # # plt.show()
        # plt.imshow(GAM, cmap='gray', alpha=0.8)
        # plt.show()

        if img.ndim == 2:
            img = img[np.newaxis]
        else:
            img = img.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)

        normalize = torchvision.transforms.Normalize(mean=[0.39895892, 0.42411209, 0.40939609], std=[0.19080092, 0.18127358, 0.19950577])
        img = normalize(torch.from_numpy(img))

        return img, GAM, numCar

    def __len__(self):
        return len(self.imgs)
