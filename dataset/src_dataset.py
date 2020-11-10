import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
from glob import glob
import xml.dom.minidom

from structures import Boxes, Instances


def parse_xml(xml_path):
    dom = xml.dom.minidom.parse(xml_path)
    xml_root = dom.documentElement
    width = xml_root.getElementsByTagName('width')[0].firstChild.data
    height = xml_root.getElementsByTagName('height')[0].firstChild.data

    meta_info = {'width': int(width), 'height': int(height)}

    K = xml_root.getElementsByTagName('object')
    xmin = xml_root.getElementsByTagName('xmin')
    xmax = xml_root.getElementsByTagName('xmax')
    ymin = xml_root.getElementsByTagName('ymin')
    ymax = xml_root.getElementsByTagName('ymax')

    boxes = []
    for k in range(len(K)):
        x1 = float(xmin[k].firstChild.data)
        y1 = float(ymin[k].firstChild.data)
        x2 = float(xmax[k].firstChild.data)
        y2 = float(ymax[k].firstChild.data)
        boxes.append([x1, y1, x2, y2])

    return torch.tensor(boxes).float(), torch.zeros(len(boxes)), meta_info


class SRCDataset(Dataset):
    def __init__(self, root, train=True):
        self.root = root
        self.train = train
        self.imgs = glob(os.path.join(root, 'img/*.jpeg'))[:2]
        self.xmls = [i.replace('img', 'label').replace('.jpeg', '.xml') for i in self.imgs]

        # simple aug
        self.resize = 512
        self.aug = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.40789654, 0.44719302, 0.47026115),
                                 std=(0.28863828, 0.27408164, 0.27809835))
        ])

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert('RGB')
        boxes, gt_classes, meta_info = parse_xml(self.xmls[index])

        meta_info['file_name'] = self.imgs[index]
        height_ratio = self.resize / meta_info['height']
        width_ratio = self.resize / meta_info['width']
        boxes[:, [0, 2]] *= width_ratio
        boxes[:, [1, 3]] *= height_ratio
        meta_info['resize'] = self.resize  # into rectangle

        img = self.aug(img)

        return img, boxes, gt_classes, meta_info

    def __len__(self):
        return len(self.imgs)

    def collate_fn(self, batch):
        imgs_list, boxes_list, gt_classes_list, meta_info_list = zip(*batch)
        assert len(imgs_list) == len(boxes_list) == len(meta_info_list)

        batched_inputs = []
        for i in range(len(imgs_list)):
            _dict = {}
            _dict['file_name'] = meta_info_list[i]['file_name']
            _dict['height'] = meta_info_list[i]['height']
            _dict['width'] = meta_info_list[i]['width']

            _dict['image'] = imgs_list[i]
            _dict['instances'] = Instances(image_size=(meta_info_list[i]['resize'], meta_info_list[i]['resize']))
            _dict['instances'].gt_boxes = Boxes(boxes_list[i])
            _dict['instances'].gt_classes = gt_classes_list[i].long()

            batched_inputs.append(_dict)
        return batched_inputs


if __name__ == '__main__':
    ds = SRCDataset('../../cascade-rcnn_lx/data/SRC/')
    dl = torch.utils.data.DataLoader(ds, batch_size=2, collate_fn=ds.collate_fn)

    batched_inputs = next(iter(dl))
