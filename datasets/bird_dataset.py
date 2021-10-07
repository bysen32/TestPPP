import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pdb
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


DATAPATH = './CUB_200_2011'
image_path = {}
image_label = {}

class BirdDataset(Dataset):
    def __init__(self, phase="train", **kwargs):
        assert phase in ['train', 'val', 'test']
        self.__dict__.update(kwargs)
        self.phase = phase
        self.num_classes = 200
        self.image_id = []

        with open(os.path.join(DATAPATH, 'images.txt')) as f:
            for line in f.readlines():
                id, path = line.strip().split(' ')
                image_path[id] = path
        
        with open(os.path.join(DATAPATH, 'image_class_labels.txt')) as f:
            for line in f.readlines():
                id, label = line.strip().split(' ')
                image_label[id] = int(label)
        
        with open(os.path.join(DATAPATH, 'train_test_split.txt')) as f:
            for line in f.readlines():
                image_id, is_train = line.strip().split(' ')
                is_train = int(is_train)

                if self.phase == 'train' and is_train:
                    self.image_id.append(image_id)
                if self.phase in ('val', 'test') and not is_train:
                    self.image_id.append(image_id)
        
        self.transform = get_transform(self.image_size, self.phase)

        if self.dev_mode:
            self.image_id = self.image_id[:50]
    
    def __getitem__(self, item):
        image_id = self.image_id[item]

        image = Image.open(os.path.join(DATAPATH, 'images', image_path[image_id])).convert('RGB')
        image = self.transform(image)

        return {"image_name": image_id, "image": image, "label": image_label[image_id]-1}

    def __len__(self):
        return len(self.image_id)


##################################
# transform in dataset
##################################
def get_transform(resize, phase='train'):
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize(size=(int(resize / 0.875), int(resize / 0.875))),
            transforms.RandomCrop(resize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[ 0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(size=(int(resize / 0.875), int(resize / 0.875))),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

if __name__ == '__main__':
    ds = BirdDataset('train')
    for i in range(10):
        image, label = ds[i]
        print(image.shape, label)
