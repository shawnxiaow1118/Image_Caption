import nltk
import pickle
from collections import Counter
# have to install coco dataset from https://github.com/pdollar/coco.git
from pycocotools.coco import COCO
import argparse

class Vocabulary(object):
    """ simple one hot encoder for word"""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        
    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
            
    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unknown>']
        return self.word2idx[word]
    
    def __len__(self):
        return len(self.word2idx)



def build_vocab(json, threshold=2):
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)
        
        if i%1000 == 0:
            print("{}/{} tokenized the captions".format(i, len(ids)))
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unknown>')
    
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab



def main(args):
    vocab = build_vocab(json=args.caption_path, threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("total vocabulary size is {}".format(len(vocab)))
    print("vocab saved to  {}".format(vocab_path))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path',type=str,
                       default='../data/captions_train2014.json', help='path for training annotations')
    parser.add_argument('--vocab_path',type=str,
                       default='../data/vocab.pkl', help='path for saveing vocabulary')
    parser.add_argument('--caption_path',type=int,
                       default=2, help='minimum word count threshold')
    
    args = parser.parse_args()
    main(args)
    
    