from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import torch
import torch.nn as nn
import numpy as np 
import pickle
import os

from data_loader import get_loader
from build_vocab import Vocabulary
from model import Encoder, Decoder
from torchvision import transforms

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

trans = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])

with open("../data/vocab.pkl", 'rb') as f:
    vocab = pickle.load(f)
dataloader = get_loader("../data/resized/", "../data/annotations/captions_train2014.json",
         vocab, trans,128, shuffle=True)

encoder = Encoder(256)
decoder = Decoder(256,512, len(vocab),1)

if torch.cuda.is_available():
    encoder.cuda()
    decoder.cuda()

    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001)
    
    total_step = len(dataloader)
    for epoch in range(5):
        for i, (images, captions, lengths) in enumerate(dataloader):
            images = to_var(images, volatile=True)
            captions = to_var(captions)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            decoder.zero_grad()
            encoder.zero_grad()
            features = encoder(images)
            
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            if i%10 == 0:
                print("Epoch {} step {}, Loss: {}, Perplexity: {}".format(epoch, i, loss.data[0], np.exp(loss.data[0])))
            if (i+1)%1000 == 0:
                torch.save(decoder.state_dict(), os.path.join("../output/",'decoder-{}-{}.pkl'.format(epoch, i+1)))
                torch.save(encoder.state_dict(), os.path.join("../output/",'encoder-{}-{}.pkl'.format(epoch, i+1)))
                
            