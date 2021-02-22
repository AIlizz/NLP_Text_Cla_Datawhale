# NLP实践-中文预训练模型泛化能力挑战赛（文本分类，bert）专题 baseline解析

比赛地址：https://tianchi.aliyun.com/s/3bd272d942f97725286a8e44f40f3f74<br>
开源内容：https://github.com/datawhalechina/team-learning-nlp/tree/master/PretrainModelsGeneralization

## baseline的文件结构
|- bert_pretrain_model #预训练模型<br>
&emsp;|- config.json<br>
&emsp;|- pytorch_model.bin<br>
&emsp;|- vocab.txt<br>
|- submission # 提交模型<br>
&emsp;|- Dockerfile # docker push的时候的必备文件，是一些docker的命令<br>
&emsp;|- ocemotion_predict.json # 模型训练train完之后，inference就可以预测得到的结果<br>
&emsp;|- ocnli_predict.json # 模型训练train完之后，inference就可以预测得到的结果<br>
&emsp;|- tnews_predict.json # 模型训练train完之后，inference就可以预测得到的结果<br>
&emsp;|- run.sh # 在用docker提交时候，如果需要运行一些程序，可以写在run.sh<br>
&emsp;|- result.zip # 在用docker提交结果的时候，首先需要将以上三个json结构压缩成result.zip，然后在用docker build和push<br>
|- tianchi_datasets # 数据集<br>
&emsp;|- OCEMOTION<br>
&emsp;&emsp;|- total.csv # trian数据集，下同<br>
&emsp;&emsp;|- test.csv # test数据集，下同<br>
&emsp;|- OCNLI<br>
&emsp;&emsp;|- total.csv<br>
&emsp;&emsp;|- test.csv<br>
&emsp;|- TNEWS<br>
&emsp;&emsp;|- total.csv<br>
&emsp;&emsp;|- test.csv<br>
&emsp;|- label.json # 运行generate_data.py之后，可以得到每个分类都有什么label<br>
&emsp;|- label_weight.json # 运行generate_data.py之后，可以得到每个分类的每个label的权重？？？？？<br>
|- calculate_loss.py # 计算loss<br>
|- data_generator.py # 数据处理？？？<br>
|- generate_data.py # 将原始数据的total切分为train和test,并将数据都转化为json数据<br>
|- inference.py # 预测数据<br>
|- net.py # 神经网络模型<br>
|- train.py # 训练<br>
|- utils.py # 工具<br>

## generate_data的解析
```python
# 导入库
import json
from collections import defaultdict
from math import log

# 定义将数据分开
def split_dataset(dev_data_cnt=5000):
    '''
    dev_data_cnts是dev数据集的数目
    '''
    pass
                            
def print_one_data(path, name, print_content=False):
    '''
    打开文件，用json读入，打印文本有多少行
    '''
    pass

def generate_data():
    '''
    
    '''
    label_set = dict()
    label_cnt_set = dict()
    for e in ['TNEWS', 'OCNLI', 'OCEMOTION']:
        #统计每个数据集的total.csv中的所有的label，放在label_set
        #统计每个数据集的total.csv中的所有的{label：label出现的次数}，放在label_cnt_set
    for k in label_set:
        label_set[k] = sorted(list(label_set[k])) # 对label进行排序
    for k, v in label_set.items(): # 打印
        print(k, v)
    with open('./tianchi_datasets/label.json', 'w',encoding='utf-8') as fw: # 把label_set写到labe.json中
        fw.write(json.dumps(label_set))
    label_weight_set = dict()
    for k in label_set: # 计算每个数据集中的每个label的权重，权重计算公式log(total_weight / e)，出现的越少，权重越大
        label_weight_set[k] = [label_cnt_set[k][e] for e in label_set[k]]
        total_weight = sum(label_weight_set[k])
        label_weight_set[k] = [log(total_weight / e) for e in label_weight_set[k]]
    for k, v in label_weight_set.items(): # 打印
        print(k, v)
    with open('./tianchi_datasets/label_weights.json', 'w',encoding='utf-8') as fw: # 将权重label_weight_set写到label_weights.json中
        fw.write(json.dumps(label_weight_set))
    
    for e in ['TNEWS', 'OCNLI', 'OCEMOTION']: # 将从total.csv中分解出来的dev.csv和train.csv转化为json文件
        for name in ['dev', 'train']:
            with open('./tianchi_datasets/' + e + '/' + name + '.csv',encoding='utf-8') as fr:
                with open('./tianchi_datasets/' + e + '/' + name + '.json', 'w',encoding='utf-8') as fw:
                    json_dict = dict()
                    for line in fr:
                        tmp_list = line.strip().split('\t')
                        json_dict[tmp_list[0]] = dict()
                        json_dict[tmp_list[0]]['s1'] = tmp_list[1]
                        if e == 'OCNLI':
                            json_dict[tmp_list[0]]['s2'] = tmp_list[2]
                            json_dict[tmp_list[0]]['label'] = tmp_list[3]
                        else:
                            json_dict[tmp_list[0]]['label'] = tmp_list[2]
                    fw.write(json.dumps(json_dict))
    
    for e in ['TNEWS', 'OCNLI', 'OCEMOTION']: # 将文件打印出来
        for name in ['dev', 'train']:
            cur_path = './tianchi_datasets/' + e + '/' + name + '.json'
            data_name = e + '_' + name
            print_one_data(cur_path, data_name)
            
    print_one_data('./tianchi_datasets/label.json', 'label_set') # 将label_set.json文件打印
    
if __name__ == '__main__':
    print('-------------------------------start-----------------------------------')
    split_dataset(dev_data_cnt=3000)
    generate_data()
    print('-------------------------------finish-----------------------------------')
```
## net 网络解析
```python
# 导入库函数，采用torch和transformers
import torch
from torch import nn # torch的神经网络
from transformers import BertModel # 从transformers导入BertModel

class Net(nn.Module):
    def __init__(self, bert_model):
        super(Net, self).__init__()
self.bert = bert_model
        self.atten_layer = nn.Linear(768, 16) # attention层是全连接
        self.softmax_d1 = nn.Softmax(dim=1) # Softmax层
        self.dropout = nn.Dropout(0.2) # Dropout层
        self.OCNLI_layer = nn.Linear(768, 16 * 3) # bert输出之后，接全连接，输出为16*3，共3个类别，16什么意思？
        self.OCEMOTION_layer = nn.Linear(768, 16 * 7) # bert输出之后，接全连接，输出为16*7，共7个类别，16什么意思？
        self.TNEWS_layer = nn.Linear(768, 16 * 15) # bert输出之后，接全连接，输出为16*15，共15个类别，16什么意思？

    def forward(self, input_ids, ocnli_ids, ocemotion_ids, tnews_ids, token_type_ids=None, attention_mask=None):
        '''
        可以参考Tranformers.BertModel的用法：https://blog.csdn.net/claroja/article/details/108492518
        Bert出来之后，开始构建不同的计算
        这边可以大改
        '''
        cls_emb = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0][:, 0, :].squeeze(1)
        if ocnli_ids.size()[0] > 0:
            attention_score = self.atten_layer(cls_emb[ocnli_ids, :])
            attention_score = self.dropout(self.softmax_d1(attention_score).unsqueeze(1))
            ocnli_value = self.OCNLI_layer(cls_emb[ocnli_ids, :]).contiguous().view(-1, 16, 3)
            ocnli_out = torch.matmul(attention_score, ocnli_value).squeeze(1)
        else:
            ocnli_out = None
        if ocemotion_ids.size()[0] > 0:
            attention_score = self.atten_layer(cls_emb[ocemotion_ids, :])
            attention_score = self.dropout(self.softmax_d1(attention_score).unsqueeze(1))
            ocemotion_value = self.OCEMOTION_layer(cls_emb[ocemotion_ids, :]).contiguous().view(-1, 16, 7)
            ocemotion_out = torch.matmul(attention_score, ocemotion_value).squeeze(1)
        else:
            ocemotion_out = None
        if tnews_ids.size()[0] > 0:
            attention_score = self.atten_layer(cls_emb[tnews_ids, :])
            attention_score = self.dropout(self.softmax_d1(attention_score).unsqueeze(1))
            tnews_value = self.TNEWS_layer(cls_emb[tnews_ids, :]).contiguous().view(-1, 16, 15)
            tnews_out = torch.matmul(attention_score, tnews_value).squeeze(1)
        else:
            tnews_out = None
        return ocnli_out, ocemotion_out, tnews_out
```

## calculate_loss的解析
```python

# 导入库函数
import torch
from torch import nn
import numpy as np
from math import exp, log


class Calculate_loss():
    def __init__(self, label_dict, weighted=False, tnews_weights=None, ocnli_weights=None, ocemotion_weights=None):
        '''
        求损失
        label_dict标签字典
        weighted是否需要考虑权重
        tnews_weights权重
        ocnli_weights权重
        ocemotion_weights权重
        '''
        self.weighted = weighted
        if weighted:
            self.tnews_loss = nn.CrossEntropyLoss(tnews_weights)
            self.ocnli_loss = nn.CrossEntropyLoss(ocnli_weights)
            self.ocemotion_loss = nn.CrossEntropyLoss(ocemotion_weights)
        else:
            self.loss = nn.CrossEntropyLoss()
        # 根据label_dict构建label2idx和idx2label
        self.label2idx = dict()
        self.idx2label = dict()
        for key in ['TNEWS', 'OCNLI', 'OCEMOTION']:
            self.label2idx[key] = dict()
            self.idx2label[key] = dict()
            for i, e in enumerate(label_dict[key]):
                self.label2idx[key][e] = i
                self.idx2label[key][i] = e
    
    def idxToLabel(self, key, idx):
        return self.idx2Label[key][idx]
    
    def labelToIdx(self, key, label):
        return self.label2idx[key][label]
    
    def compute(self, tnews_pred, ocnli_pred, ocemotion_pred, tnews_gold, ocnli_gold, ocemotion_gold):
        '''
        计算损失，采用交叉熵nn.CrossEntropyLoss计算损失，三个损失采用相加的方式
        '''
        res = 0
        if tnews_pred != None:
            res += self.tnews_loss(tnews_pred, tnews_gold) if self.weighted else self.loss(tnews_pred, tnews_gold)
        if ocnli_pred != None:
            res += self.ocnli_loss(ocnli_pred, ocnli_gold) if self.weighted else self.loss(ocnli_pred, ocnli_gold)
        if ocemotion_pred != None:
            res += self.ocemotion_loss(ocemotion_pred, ocemotion_gold) if self.weighted else self.loss(ocemotion_pred, ocemotion_gold)
        return res

    def compute_dtp(self, tnews_pred, ocnli_pred, ocemotion_pred, tnews_gold, ocnli_gold, ocemotion_gold, tnews_kpi=0.1, ocnli_kpi=0.1, ocemotion_kpi=0.1, y=0.5):
        '''
                
        '''
        res = 0
        if tnews_pred != None:
            res += self.tnews_loss(tnews_pred, tnews_gold) * self._calculate_weight(tnews_kpi, y) if self.weighted else self.loss(tnews_pred, tnews_gold) * self._calculate_weight(tnews_kpi, y)
        if ocnli_pred != None:
            res += self.ocnli_loss(ocnli_pred, ocnli_gold) * self._calculate_weight(ocnli_kpi, y) if self.weighted else self.loss(ocnli_pred, ocnli_gold) * self._calculate_weight(ocnli_kpi, y)
        if ocemotion_pred != None:
            res += self.ocemotion_loss(ocemotion_pred, ocemotion_gold) * self._calculate_weight(ocemotion_kpi, y) if self.weighted else self.loss(ocemotion_pred, ocemotion_gold) * self._calculate_weight(ocemotion_kpi, y)
        return res

    
    def correct_cnt(self, tnews_pred, ocnli_pred, ocemotion_pred, tnews_gold, ocnli_gold, ocemotion_gold):
        good_nb = 0
        total_nb = 0
        if tnews_pred != None:
            tnews_val = torch.argmax(tnews_pred, axis=1)
            for i, e in enumerate(tnews_gold):
                if e == tnews_val[i]:
                    good_nb += 1
                total_nb += 1
        if ocnli_pred != None:
            ocnli_val = torch.argmax(ocnli_pred, axis=1)
            for i, e in enumerate(ocnli_gold):
                if e == ocnli_val[i]:
                    good_nb += 1
                total_nb += 1
        if ocemotion_pred != None:
            ocemotion_val = torch.argmax(ocemotion_pred, axis=1)
            for i, e in enumerate(ocemotion_gold):
                if e == ocemotion_val[i]:
                    good_nb += 1
                total_nb += 1
        return good_nb, total_nb

    def correct_cnt_each(self, tnews_pred, ocnli_pred, ocemotion_pred, tnews_gold, ocnli_gold, ocemotion_gold):
        good_ocnli_nb = 0
        good_ocemotion_nb = 0
        good_tnews_nb = 0
        total_ocnli_nb = 0
        total_ocemotion_nb = 0
        total_tnews_nb = 0
        if tnews_pred != None:
            tnews_val = torch.argmax(tnews_pred, axis=1)
            for i, e in enumerate(tnews_gold):
                if e == tnews_val[i]:
                    good_tnews_nb += 1
                total_tnews_nb += 1
        if ocnli_pred != None:
            ocnli_val = torch.argmax(ocnli_pred, axis=1)
            for i, e in enumerate(ocnli_gold):
                if e == ocnli_val[i]:
                    good_ocnli_nb += 1
                total_ocnli_nb += 1
        if ocemotion_pred != None:
            ocemotion_val = torch.argmax(ocemotion_pred, axis=1)
            for i, e in enumerate(ocemotion_gold):
                if e == ocemotion_val[i]:
                    good_ocemotion_nb += 1
                total_ocemotion_nb += 1
        return good_tnews_nb, good_ocnli_nb, good_ocemotion_nb, total_tnews_nb, total_ocnli_nb, total_ocemotion_nb
    
    def collect_pred_and_gold(self, pred, gold):
        if pred == None or gold == None:
            p, g = [], []
        else:
            p, g = np.array(torch.argmax(pred, axis=1).cpu()).tolist(), np.array(gold.cpu()).tolist()
        return p, g

    def _calculate_weight(self, kpi, y):
        kpi = max(0.1, kpi)
        kpi = min(0.99, kpi)
        w = -1 * ((1 - kpi) ** y) * log(kpi)
        return w
```

## train的解析


## inference的解析
