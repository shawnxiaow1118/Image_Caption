import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from build_vocab import Vocabulary
from PIL import Image
from pycocotools.coco import COCO



class DataSet(data.Dataset):
    def __init__(self, path, json, vocab, transform=None):
        self.path = path
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform
        
    def __getitem__(self, index):
        ann_id = self.ids[index]
        coco = self.coco
        vocab = self.vocab
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        img_name = coco.loadImgs(img_id)[0]['file_name']
        
        image = Image.open(os.path.join(self.path,img_name)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target
    
    def __len__(self):
        return len(self.ids)



def collate_fn(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)
    images = torch.stack(images, 0)
    
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i,:end] = cap[:end]
    return images, targets, lengths
    


def get_loader(path, json, vocab, transform, batch_size, shuffle):
    coco = DataSet(path=path,json=json,vocab=vocab,transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             collate_fn=collate_fn)
    return data_loader