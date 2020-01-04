import torch
import torch.nn as nn

from torchtext import data
from torchtext.data import Iterator, BucketIterator, TabularDataset
from torchtext.vocab import Vectors, GloVe

class FastText(nn.Module):
    def __init__(self, vocab, vec_dim, label_size, hidden_size):
        super(FastText, self).__init__()
        #创建embedding
        self.embed = nn.Embedding(len(vocab), vec_dim)
        # 若使用预训练的词向量，需在此处指定预训练的权重
        self.embed.weight.data.copy_(vocab.vectors)
        self.embed.weight.requires_grad = True
        self.fc = nn.Sequential(
            nn.Linear(vec_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, label_size))

        self.vocab = vocab

    def forward(self, x):
        # x = self.embed(x)
        out = self.fc(torch.mean(x, dim=1))
        return out

def get_data_iter(train_csv="/home/dengerqiang/Documents/WORK/DataSets/ag_news_csv/train.csv",
                  test_csv="/home/dengerqiang/Documents/WORK/DataSets/ag_news_csv/test.csv",
                  fix_length=100, batch_size=64):
    TEXT = data.Field(sequential=True, lower=True, fix_length=fix_length, batch_first=True)
    LABEL = data.Field(sequential=False, use_vocab=False)
    train_fields = [("label", LABEL), ("title", None), ("text", TEXT)]
    train = TabularDataset(path=train_csv, format="csv", fields=train_fields, skip_header=True)
    train_iter = BucketIterator(train, batch_size=batch_size, device=-1, sort_key=lambda x: len(x.text),
                                sort_within_batch=False, repeat=False)
    test_fields = [("label", LABEL), ("title", None), ("text", TEXT)]
    test = TabularDataset(path=test_csv, format="csv", fields=test_fields, skip_header=True)
    test_iter = Iterator(test, batch_size=batch_size, device=-1, sort=False, sort_within_batch=False, repeat=False)

    TEXT.build_vocab(train, vectors=GloVe(name='6B',dim=300))
    vocab = TEXT.vocab
    return train_iter, test_iter, vocab

def get_model(pre_trained = False):
    train_csv = "/home/dengerqiang/Documents/WORK/DataSets/ag_news_csv/train.csv"
    test_csv = "/home/dengerqiang/Documents/WORK/DataSets/ag_news_csv/test.csv"
    word2vec_dir = "glove.6B.300d.txt"  # 训练好的词向量文件,写成相对路径好像会报错
    sentence_max_size = 100  # 每篇文章的最大词数量
    net_dir = "saved_model/ag_fasttext_model.pth.tar"
    batch_size = 64
    epoch = 10  # 迭代次数
    emb_dim = 300  # 词向量维度
    lr = 0.001
    hidden_size = 200
    label_size = 4
    device = torch.device('cuda:1')

    train_iter, test_iter, vocab = get_data_iter(train_csv, test_csv, sentence_max_size)
    # 定义模型
    model = FastText(vocab=vocab, vec_dim=emb_dim, label_size=label_size, hidden_size=hidden_size)

    if pre_trained:
        model_dict = torch.load(net_dir)
        model.load_state_dict(model_dict['state_dict'])

    return model, train_iter, test_iter
