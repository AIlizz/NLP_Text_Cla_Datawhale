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
        loss上在compute函数的基础上乘以self._calculate_weight(tnews_kpi, y)    
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
        '''
        统计预测正确的数量，及样本数量(3个数据集一起计算)
        '''
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
        '''
        三个数据集分开统计预测正确的数量，及样本数量
        '''
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
        '''
        将pred概率转化为pred类别，并将gold转化为???
        '''
        if pred == None or gold == None:
            p, g = [], []
        else:
            p, g = np.array(torch.argmax(pred, axis=1).cpu()).tolist(), np.array(gold.cpu()).tolist()
        return p, g

    def _calculate_weight(self, kpi, y):
        '''
        计算权重，？？？
        '''
        kpi = max(0.1, kpi)
        kpi = min(0.99, kpi)
        w = -1 * ((1 - kpi) ** y) * log(kpi)
        return w
```

## utils解析
```python 
# 导入函数库
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report, f1_score
from transformers import BertModel, BertTokenizer

def get_f1(l_t, l_p):
    '''
    计算宏观F1
    '''
    marco_f1_score = f1_score(l_t, l_p, average='macro')
    return marco_f1_score

def print_result(l_t, l_p):
    '''
    打印marco_f1，混淆矩阵，分类报告（P,R,F1,标签出现的次数Support,平均值）
    '''
    marco_f1_score = f1_score(l_t, l_p, average='macro')
    print(marco_f1_score)
    print(f"{'confusion_matrix':*^80}")
    print(confusion_matrix(l_t, l_p, ))
    print(f"{'classification_report':*^80}")
    print(classification_report(l_t, l_p, ))

def load_tokenizer(path_or_name):
    '''
    加载预训练模型的分词器
    '''
    return BertTokenizer.from_pretrained(path_or_name)

def load_pretrained_model(path_or_name):
    '''
    加载预训练模型
    '''
    return BertModel.from_pretrained(path_or_name)

def get_task_chinese(task_type):
    '''
    解释数据集的中文释意
    '''
    if task_type == 'ocnli':
        return '(中文原版自然语言推理)'
    elif task_type == 'ocemotion':
        return '(中文情感分类)'
    else:
        return '(今日头条新闻标题分类)'
```

## data_genetate的解析
```python
# 导入库函数
import random
import torch
from transformers import BertTokenizer

class Data_generator():
    def __init__(self, ocnli_dict, ocemotion_dict, tnews_dict, label_dict, device, tokenizer, max_len=512):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.device = device
        self.label2idx = dict()
        self.idx2label = dict()
        for key in ['TNEWS', 'OCNLI', 'OCEMOTION']:
            self.label2idx[key] = dict()
            self.idx2label[key] = dict()
            for i, e in enumerate(label_dict[key]):
                self.label2idx[key][e] = i
                self.idx2label[key][i] = e
        self.ocnli_data = dict()
        self.ocnli_data['s1'] = []
        self.ocnli_data['s2'] = []
        self.ocnli_data['label'] = []
        for k, v in ocnli_dict.items():
            self.ocnli_data['s1'].append(v['s1'])
            self.ocnli_data['s2'].append(v['s2'])
            self.ocnli_data['label'].append(self.label2idx['OCNLI'][v['label']])
        self.ocemotion_data = dict()
        self.ocemotion_data['s1'] = []
        self.ocemotion_data['label'] = []
        for k, v in ocemotion_dict.items():
            self.ocemotion_data['s1'].append(v['s1'])
            self.ocemotion_data['label'].append(self.label2idx['OCEMOTION'][v['label']])
        self.tnews_data = dict()
        self.tnews_data['s1'] = []
        self.tnews_data['label'] = []
        for k, v in tnews_dict.items():
            self.tnews_data['s1'].append(v['s1'])
            self.tnews_data['label'].append(self.label2idx['TNEWS'][v['label']])
        self.reset()
    def reset(self):
        self.ocnli_ids = list(range(len(self.ocnli_data['s1'])))
        self.ocemotion_ids = list(range(len(self.ocemotion_data['s1'])))
        self.tnews_ids = list(range(len(self.tnews_data['s1'])))
        random.shuffle(self.ocnli_ids)
        random.shuffle(self.ocemotion_ids)
        random.shuffle(self.tnews_ids)
    def get_next_batch(self, batchSize=64):
        ocnli_len = len(self.ocnli_ids)
        ocemotion_len = len(self.ocemotion_ids)
        tnews_len = len(self.tnews_ids)
        total_len = ocnli_len + ocemotion_len + tnews_len
        if total_len == 0:
            return None
        elif total_len > batchSize:
            if ocnli_len > 0:
                ocnli_tmp_len = int((ocnli_len / total_len) * batchSize)
                ocnli_cur = self.ocnli_ids[:ocnli_tmp_len]
                self.ocnli_ids = self.ocnli_ids[ocnli_tmp_len:]
            if ocemotion_len > 0:
                ocemotion_tmp_len = int((ocemotion_len / total_len) * batchSize)
                ocemotion_cur = self.ocemotion_ids[:ocemotion_tmp_len]
                self.ocemotion_ids = self.ocemotion_ids[ocemotion_tmp_len:]
            if tnews_len > 0:
                tnews_tmp_len = batchSize - len(ocnli_cur) - len(ocemotion_cur)
                tnews_cur = self.tnews_ids[:tnews_tmp_len]
                self.tnews_ids = self.tnews_ids[tnews_tmp_len:]
        else:
            ocnli_cur = self.ocnli_ids
            self.ocnli_ids = []
            ocemotion_cur = self.ocemotion_ids
            self.ocemotion_ids = []
            tnews_cur = self.tnews_ids
            self.tnews_ids = []
        max_len = self._get_max_total_len(ocnli_cur, ocemotion_cur, tnews_cur)
        input_ids = []
        token_type_ids = []
        attention_mask = []
        ocnli_gold = None
        ocemotion_gold = None
        tnews_gold = None
        if len(ocnli_cur) > 0:
            flower = self.tokenizer([self.ocnli_data['s1'][idx] for idx in ocnli_cur], [self.ocnli_data['s2'][idx] for idx in ocnli_cur], add_special_tokens=True, max_length=max_len, padding='max_length', return_tensors='pt', truncation=True)
            input_ids.append(flower['input_ids'])
            token_type_ids.append(flower['token_type_ids'])
            attention_mask.append(flower['attention_mask'])
            ocnli_gold = torch.tensor([self.ocnli_data['label'][idx] for idx in ocnli_cur]).to(self.device)
        if len(ocemotion_cur) > 0:
            flower = self.tokenizer([self.ocemotion_data['s1'][idx] for idx in ocemotion_cur], add_special_tokens=True, max_length=max_len, padding='max_length', return_tensors='pt', truncation=True)
            input_ids.append(flower['input_ids'])
            token_type_ids.append(flower['token_type_ids'])
            attention_mask.append(flower['attention_mask'])
            ocemotion_gold = torch.tensor([self.ocemotion_data['label'][idx] for idx in ocemotion_cur]).to(self.device)
        if len(tnews_cur) > 0:
            flower = self.tokenizer([self.tnews_data['s1'][idx] for idx in tnews_cur], add_special_tokens=True, max_length=max_len, padding='max_length', return_tensors='pt', truncation=True)
            input_ids.append(flower['input_ids'])
            token_type_ids.append(flower['token_type_ids'])
            attention_mask.append(flower['attention_mask'])
            tnews_gold = torch.tensor([self.tnews_data['label'][idx] for idx in tnews_cur]).to(self.device)
        st = 0
        ed = len(ocnli_cur)
        ocnli_tensor = torch.tensor([i for i in range(st, ed)]).to(self.device)
        st += len(ocnli_cur)
        ed += len(ocemotion_cur)
        ocemotion_tensor = torch.tensor([i for i in range(st, ed)]).to(self.device)
        st += len(ocemotion_cur)
        ed += len(tnews_cur)
        tnews_tensor = torch.tensor([i for i in range(st, ed)]).to(self.device)
        input_ids = torch.cat(input_ids, axis=0).to(self.device)
        token_type_ids = torch.cat(token_type_ids, axis=0).to(self.device)
        attention_mask = torch.cat(attention_mask, axis=0).to(self.device)
        res = dict()
        res['input_ids'] = input_ids
        res['token_type_ids'] = token_type_ids
        res['attention_mask'] = attention_mask
        res['ocnli_ids'] = ocnli_tensor
        res['ocemotion_ids'] = ocemotion_tensor
        res['tnews_ids'] = tnews_tensor
        res['ocnli_gold'] = ocnli_gold
        res['ocemotion_gold'] = ocemotion_gold
        res['tnews_gold'] = tnews_gold
        return res

    def _get_max_total_len(self, ocnli_cur, ocemotion_cur, tnews_cur):
        res = 1
        for idx in ocnli_cur:
            res = max(res, 3 + len(self.ocnli_data['s1'][idx]) + len(self.ocnli_data['s2'][idx]))
        for idx in ocemotion_cur:
            res = max(res, 2 + len(self.ocemotion_data['s1'][idx]))
        for idx in tnews_cur:
            res = max(res, 2 + len(self.tnews_data['s1'][idx]))
        return min(res, self.max_len)
```

## train的解析
```python

```


## inference的解析
