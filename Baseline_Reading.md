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
&emsp;|- label_weight.json # 运行generate_data.py之后，可以得到每个分类的每个label的权重<br>
|- calculate_loss.py # 计算loss<br>
|- data_generator.py # 将csv的数据处理成json数据，方便模型调用<br>
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
        self.max_len = max_len # 每个句子最长的长度
        self.tokenizer = tokenizer # 分词器
        self.device = device # 计算设备
        self.label2idx = dict() # 字典 标签：索引
        self.idx2label = dict() # 字典 索引：标签
        for key in ['TNEWS', 'OCNLI', 'OCEMOTION']: # 利用label_dict构建三个数据集的label index的字典
            self.label2idx[key] = dict()
            self.idx2label[key] = dict()
            for i, e in enumerate(label_dict[key]):
                self.label2idx[key][e] = i
                self.idx2label[key][i] = e
        # 将ocnli_data的数据转化为字典self.ocnli_data{'s1':[],'s2':[],'label':[]}
        self.ocnli_data = dict()
        self.ocnli_data['s1'] = []
        self.ocnli_data['s2'] = []
        self.ocnli_data['label'] = []
        for k, v in ocnli_dict.items():
            self.ocnli_data['s1'].append(v['s1'])
            self.ocnli_data['s2'].append(v['s2'])
            self.ocnli_data['label'].append(self.label2idx['OCNLI'][v['label']])
        # 将ocemotion_data的数据转化为字典self.ocemotion_data{'s1':[],'label':[]}
        self.ocemotion_data = dict()
        self.ocemotion_data['s1'] = []
        self.ocemotion_data['label'] = []
        for k, v in ocemotion_dict.items():
            self.ocemotion_data['s1'].append(v['s1'])
            self.ocemotion_data['label'].append(self.label2idx['OCEMOTION'][v['label']])
        # 将tnews_data的数据转化为字典self.tnews_data{'s1':[],'label':[]}
        self.tnews_data = dict()
        self.tnews_data['s1'] = []
        self.tnews_data['label'] = []
        for k, v in tnews_dict.items():
            self.tnews_data['s1'].append(v['s1'])
            self.tnews_data['label'].append(self.label2idx['TNEWS'][v['label']])
        # 构建self.ocnli_ids，self.ocemotion_ids，self.tnews_ids三个list, 分别包含0到各个数据集长度-1的整数，并已打乱，方便后续取用数据
        self.reset()
    def reset(self):
        '''
        构建self.ocnli_ids，self.ocemotion_ids，self.tnews_ids三个list, 分别包含0到各个数据集长度-1的整数，并已打乱，方便后续取用数据
        '''
        self.ocnli_ids = list(range(len(self.ocnli_data['s1'])))
        self.ocemotion_ids = list(range(len(self.ocemotion_data['s1'])))
        self.tnews_ids = list(range(len(self.tnews_data['s1'])))
        random.shuffle(self.ocnli_ids)
        random.shuffle(self.ocemotion_ids)
        random.shuffle(self.tnews_ids)
    def get_next_batch(self, batchSize=64):
        '''
        从数据中取出next_batch
        '''
        ocnli_len = len(self.ocnli_ids)
        ocemotion_len = len(self.ocemotion_ids)
        tnews_len = len(self.tnews_ids)
        total_len = ocnli_len + ocemotion_len + tnews_len
        if total_len == 0:
            return None # 如果取完，就返回None
        elif total_len > batchSize:
            if ocnli_len > 0:
                ocnli_tmp_len = int((ocnli_len / total_len) * batchSize) # 按照比例取用数据
                ocnli_cur = self.ocnli_ids[:ocnli_tmp_len] # 取出数据
                self.ocnli_ids = self.ocnli_ids[ocnli_tmp_len:] # 将取出的数据去掉
            if ocemotion_len > 0:
                ocemotion_tmp_len = int((ocemotion_len / total_len) * batchSize) # 同上
                ocemotion_cur = self.ocemotion_ids[:ocemotion_tmp_len]
                self.ocemotion_ids = self.ocemotion_ids[ocemotion_tmp_len:]
            if tnews_len > 0:
                tnews_tmp_len = batchSize - len(ocnli_cur) - len(ocemotion_cur) # 同上
                tnews_cur = self.tnews_ids[:tnews_tmp_len]
                self.tnews_ids = self.tnews_ids[tnews_tmp_len:]
        else: # 如果不足一个batch_size,就将剩下所有的数据取出
            ocnli_cur = self.ocnli_ids
            self.ocnli_ids = []
            ocemotion_cur = self.ocemotion_ids
            self.ocemotion_ids = []
            tnews_cur = self.tnews_ids
            self.tnews_ids = []
        # 找到语句的最长长度
        max_len = self._get_max_total_len(ocnli_cur, ocemotion_cur, tnews_cur)
        # 采用self.tokenizer对预料进行分词，三个数据集分词结果均放在input_ids，token_type_ids，attention_mask
        # ocnli_gold，ocemotion_gold，tnews_gold是标签
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
        # 构造ocnli_tensor，ocemotion_tensor，tnews_tensor记录从数据集中取出来的数据的相对位置
        st = 0
        ed = len(ocnli_cur)
        ocnli_tensor = torch.tensor([i for i in range(st, ed)]).to(self.device)
        st += len(ocnli_cur)
        ed += len(ocemotion_cur)
        ocemotion_tensor = torch.tensor([i for i in range(st, ed)]).to(self.device)
        st += len(ocemotion_cur)
        ed += len(tnews_cur)
        tnews_tensor = torch.tensor([i for i in range(st, ed)]).to(self.device)
        
        # 将input_ids，token_type_ids，attention_mask数据按行合并
        input_ids = torch.cat(input_ids, axis=0).to(self.device)
        token_type_ids = torch.cat(token_type_ids, axis=0).to(self.device)
        attention_mask = torch.cat(attention_mask, axis=0).to(self.device)
        
        # 将input_ids，token_type_ids，attention_mask，ocnli_ids，ocemotion_ids，tnews_ids，ocnli_gold，ocemotion_gold，tnews_gold放在res字典中
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
        '''
        找到该batch语料中最大的长度返回，但是当长度超过self.max_len最大的时候，可以用self.max_len将其截断
        '''
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
# 导入库函数
import torch
from transformers import BertModel, BertTokenizer
import json
from utils import get_f1, print_result, load_pretrained_model, load_tokenizer
from net import Net
from data_generator import Data_generator
from calculate_loss import Calculate_loss


def train(epochs=20, batchSize=64, lr=0.0001, device='cuda:0', accumulate=True, a_step=16, load_saved=False, file_path='./saved_best.pt', use_dtp=False, pretrained_model='./bert_pretrain_model', tokenizer_model='bert-base-chinese', weighted_loss=False):
    device = device # 计算的设备
    tokenizer = load_tokenizer(tokenizer_model) # 分词器
    my_net = torch.load(file_path) if load_saved else Net(load_pretrained_model(pretrained_model)) # 导入网络
    my_net.to(device, non_blocking=True) # 将网络导入到设备中
    # 导入label_dict，{数据集：[标签集]}
    label_dict = dict()
    with open('./tianchi_datasets/label.json') as f:
        for line in f:
            label_dict = json.loads(line)
            break
    # label_weights_dict，{数据集：[标签权重]}
    label_weights_dict = dict()
    with open('./tianchi_datasets/label_weights.json') as f:
        for line in f:
            label_weights_dict = json.loads(line)
            break
    # 打开ocnli_train数据
    ocnli_train = dict()
    with open('./tianchi_datasets/OCNLI/train.json') as f:
        for line in f:
            ocnli_train = json.loads(line)
            break
    # 打开ocnli_dev数据
    ocnli_dev = dict()
    with open('./tianchi_datasets/OCNLI/dev.json') as f:
        for line in f:
            ocnli_dev = json.loads(line)
            break
    # 打开ocemotion_train数据
    ocemotion_train = dict()
    with open('./tianchi_datasets/OCEMOTION/train.json') as f:
        for line in f:
            ocemotion_train = json.loads(line)
            break
    # 打开ocemotion_dev数据
    ocemotion_dev = dict()
    with open('./tianchi_datasets/OCEMOTION/dev.json') as f:
        for line in f:
            ocemotion_dev = json.loads(line)
            break
    # 打开tnews_train数据
    tnews_train = dict()
    with open('./tianchi_datasets/TNEWS/train.json') as f:
        for line in f:
            tnews_train = json.loads(line)
            break
    # 打开tnews_dev数据
    tnews_dev = dict()
    with open('./tianchi_datasets/TNEWS/dev.json') as f:
        for line in f:
            tnews_dev = json.loads(line)
            break
    # 将数据制作成pytorch可用的数据
    train_data_generator = Data_generator(ocnli_train, ocemotion_train, tnews_train, label_dict, device, tokenizer)
    dev_data_generator = Data_generator(ocnli_dev, ocemotion_dev, tnews_dev, label_dict, device, tokenizer)
    # 导入tnews，ocnli，ocemotion三个数据集的权重
    tnews_weights = torch.tensor(label_weights_dict['TNEWS']).to(device, non_blocking=True)
    ocnli_weights = torch.tensor(label_weights_dict['OCNLI']).to(device, non_blocking=True)
    ocemotion_weights = torch.tensor(label_weights_dict['OCEMOTION']).to(device, non_blocking=True)
    # 定义损失计算方法
    loss_object = Calculate_loss(label_dict, weighted=weighted_loss, tnews_weights=tnews_weights, ocnli_weights=ocnli_weights, ocemotion_weights=ocemotion_weights)
    # 优化器
    optimizer=torch.optim.Adam(my_net.parameters(), lr=lr)
    best_dev_f1 = 0.0
    best_epoch = -1
    # 开始计算
    for epoch in range(epochs):
        my_net.train() # 指定model进入.train()状态，各个值都需要训练。
        train_loss = 0.0
        train_total = 0
        train_correct = 0
        train_ocnli_correct = 0
        train_ocemotion_correct = 0
        train_tnews_correct = 0
        train_ocnli_pred_list = []
        train_ocnli_gold_list = []
        train_ocemotion_pred_list = []
        train_ocemotion_gold_list = []
        train_tnews_pred_list = []
        train_tnews_gold_list = []
        cnt_train = 0
        while True:
            raw_data = train_data_generator.get_next_batch(batchSize) # 取出数据
            if raw_data == None:
                break
            data = dict()
            data['input_ids'] = raw_data['input_ids']
            data['token_type_ids'] = raw_data['token_type_ids']
            data['attention_mask'] = raw_data['attention_mask']
            data['ocnli_ids'] = raw_data['ocnli_ids']
            data['ocemotion_ids'] = raw_data['ocemotion_ids']
            data['tnews_ids'] = raw_data['tnews_ids']
            tnews_gold = raw_data['tnews_gold']
            ocnli_gold = raw_data['ocnli_gold']
            ocemotion_gold = raw_data['ocemotion_gold']
            if not accumulate:
                optimizer.zero_grad() # 每个batch是否都对模型参数的梯度置0，梯度清除 https://www.jianshu.com/p/c59b75f1064c
            # 输入数据，完成计算
            ocnli_pred, ocemotion_pred, tnews_pred = my_net(**data)
            # 计算损失，dtp是什么意思？？？
            if use_dtp:
                tnews_kpi = 0.1 if len(train_tnews_pred_list) == 0 else train_tnews_correct / len(train_tnews_pred_list)
                ocnli_kpi = 0.1 if len(train_ocnli_pred_list) == 0 else train_ocnli_correct / len(train_ocnli_pred_list)
                ocemotion_kpi = 0.1 if len(train_ocemotion_pred_list) == 0 else train_ocemotion_correct / len(train_ocemotion_pred_list)
                current_loss = loss_object.compute_dtp(tnews_pred, ocnli_pred, ocemotion_pred, tnews_gold, ocnli_gold,
                                                   ocemotion_gold, tnews_kpi, ocnli_kpi, ocemotion_kpi)
            else:
                current_loss = loss_object.compute(tnews_pred, ocnli_pred, ocemotion_pred, tnews_gold, ocnli_gold, ocemotion_gold)
            train_loss += current_loss.item()
            # 反向传播
            current_loss.backward()
            if accumulate and (cnt_train + 1) % a_step == 0:
                optimizer.step()
                optimizer.zero_grad()
            if not accumulate:
                optimizer.step()
            # 计算预测情况（tmp_good预测对的，tmp_total总共预测数量）
            if use_dtp:
                good_tnews_nb, good_ocnli_nb, good_ocemotion_nb, total_tnews_nb, total_ocnli_nb, total_ocemotion_nb = loss_object.correct_cnt_each(tnews_pred, ocnli_pred, ocemotion_pred, tnews_gold, ocnli_gold, ocemotion_gold)
                tmp_good = sum([good_tnews_nb, good_ocnli_nb, good_ocemotion_nb])
                tmp_total = sum([total_tnews_nb, total_ocnli_nb, total_ocemotion_nb])
                train_ocemotion_correct += good_ocemotion_nb
                train_ocnli_correct += good_ocnli_nb
                train_tnews_correct += good_tnews_nb
            else:
                tmp_good, tmp_total = loss_object.correct_cnt(tnews_pred, ocnli_pred, ocemotion_pred, tnews_gold, ocnli_gold, ocemotion_gold)
            train_correct += tmp_good
            train_total += tmp_total
            # 将XXX_pred（概率）转化为类别p, XXX_gold转化为g???
            p, g = loss_object.collect_pred_and_gold(ocnli_pred, ocnli_gold)
            train_ocnli_pred_list += p
            train_ocnli_gold_list += g
            p, g = loss_object.collect_pred_and_gold(ocemotion_pred, ocemotion_gold)
            train_ocemotion_pred_list += p
            train_ocemotion_gold_list += g
            p, g = loss_object.collect_pred_and_gold(tnews_pred, tnews_gold)
            train_tnews_pred_list += p
            train_tnews_gold_list += g
            cnt_train += 1
            #torch.cuda.empty_cache()
            # 每1000次打印一次
            if (cnt_train + 1) % 1000 == 0:
                print('[', cnt_train + 1, '- th batch : train acc is:', train_correct / train_total, '; train loss is:', train_loss / cnt_train, ']')
        # 计算完一个epoch的所有batch
        if accumulate:
            optimizer.step()
        optimizer.zero_grad()
        # 计算和打印f1
        train_ocnli_f1 = get_f1(train_ocnli_gold_list, train_ocnli_pred_list)
        train_ocemotion_f1 = get_f1(train_ocemotion_gold_list, train_ocemotion_pred_list)
        train_tnews_f1 = get_f1(train_tnews_gold_list, train_tnews_pred_list)
        train_avg_f1 = (train_ocnli_f1 + train_ocemotion_f1 + train_tnews_f1) / 3
        print(epoch, 'th epoch train average f1 is:', train_avg_f1)
        print(epoch, 'th epoch train ocnli is below:')
        print_result(train_ocnli_gold_list, train_ocnli_pred_list)
        print(epoch, 'th epoch train ocemotion is below:')
        print_result(train_ocemotion_gold_list, train_ocemotion_pred_list)
        print(epoch, 'th epoch train tnews is below:')
        print_result(train_tnews_gold_list, train_tnews_pred_list)
        
        # 将样本顺序重置
        train_data_generator.reset()
        
        #
        my_net.eval() #指定model进入.eval()状态，pytorch会自动把BN和DropOut固定住，不会取平均，而是用训练好的值。
        dev_loss = 0.0
        dev_total = 0
        dev_correct = 0
        dev_ocnli_correct = 0
        dev_ocemotion_correct = 0
        dev_tnews_correct = 0
        dev_ocnli_pred_list = []
        dev_ocnli_gold_list = []
        dev_ocemotion_pred_list = []
        dev_ocemotion_gold_list = []
        dev_tnews_pred_list = []
        dev_tnews_gold_list = []
        cnt_dev = 0
        with torch.no_grad(): # 不计算梯度，也不进行反向传播
            while True:
                raw_data = dev_data_generator.get_next_batch(batchSize)
                if raw_data == None:
                    break
                # 数据准备
                data = dict()
                data['input_ids'] = raw_data['input_ids']
                data['token_type_ids'] = raw_data['token_type_ids']
                data['attention_mask'] = raw_data['attention_mask']
                data['ocnli_ids'] = raw_data['ocnli_ids']
                data['ocemotion_ids'] = raw_data['ocemotion_ids']
                data['tnews_ids'] = raw_data['tnews_ids']
                tnews_gold = raw_data['tnews_gold']
                ocnli_gold = raw_data['ocnli_gold']
                ocemotion_gold = raw_data['ocemotion_gold']
                ocnli_pred, ocemotion_pred, tnews_pred = my_net(**data)
                # 计算loss
                if use_dtp:
                    tnews_kpi = 0.1 if len(dev_tnews_pred_list) == 0 else dev_tnews_correct / len(
                        dev_tnews_pred_list)
                    ocnli_kpi = 0.1 if len(dev_ocnli_pred_list) == 0 else dev_ocnli_correct / len(
                        dev_ocnli_pred_list)
                    ocemotion_kpi = 0.1 if len(dev_ocemotion_pred_list) == 0 else dev_ocemotion_correct / len(
                        dev_ocemotion_pred_list)
                    current_loss = loss_object.compute_dtp(tnews_pred, ocnli_pred, ocemotion_pred, tnews_gold,
                                                           ocnli_gold,
                                                           ocemotion_gold, tnews_kpi, ocnli_kpi, ocemotion_kpi)
                else:
                    current_loss = loss_object.compute(tnews_pred, ocnli_pred, ocemotion_pred, tnews_gold, ocnli_gold, ocemotion_gold)
                dev_loss += current_loss.item()
                # 计算预测准确的值，及总共的值
                if use_dtp:
                    good_tnews_nb, good_ocnli_nb, good_ocemotion_nb, total_tnews_nb, total_ocnli_nb, total_ocemotion_nb = loss_object.correct_cnt_each(
                        tnews_pred, ocnli_pred, ocemotion_pred, tnews_gold, ocnli_gold, ocemotion_gold)
                    tmp_good += sum([good_tnews_nb, good_ocnli_nb, good_ocemotion_nb])
                    tmp_total += sum([total_tnews_nb, total_ocnli_nb, total_ocemotion_nb])
                    dev_ocemotion_correct += good_ocemotion_nb
                    dev_ocnli_correct += good_ocnli_nb
                    dev_tnews_correct += good_tnews_nb
                else:
                    tmp_good, tmp_total = loss_object.correct_cnt(tnews_pred, ocnli_pred, ocemotion_pred, tnews_gold, ocnli_gold, ocemotion_gold)
                dev_correct += tmp_good
                dev_total += tmp_total
                # 将XXX_pred（概率）转化为类别p, XXX_gold转化为g???
                p, g = loss_object.collect_pred_and_gold(ocnli_pred, ocnli_gold)
                dev_ocnli_pred_list += p
                dev_ocnli_gold_list += g
                p, g = loss_object.collect_pred_and_gold(ocemotion_pred, ocemotion_gold)
                dev_ocemotion_pred_list += p
                dev_ocemotion_gold_list += g
                p, g = loss_object.collect_pred_and_gold(tnews_pred, tnews_gold)
                dev_tnews_pred_list += p
                dev_tnews_gold_list += g
                cnt_dev += 1
                #torch.cuda.empty_cache()
                #if (cnt_dev + 1) % 1000 == 0:
                #    print('[', cnt_dev + 1, '- th batch : dev acc is:', dev_correct / dev_total, '; dev loss is:', dev_loss / cnt_dev, ']')
            # 计算f1
            dev_ocnli_f1 = get_f1(dev_ocnli_gold_list, dev_ocnli_pred_list)
            dev_ocemotion_f1 = get_f1(dev_ocemotion_gold_list, dev_ocemotion_pred_list)
            dev_tnews_f1 = get_f1(dev_tnews_gold_list, dev_tnews_pred_list)
            dev_avg_f1 = (dev_ocnli_f1 + dev_ocemotion_f1 + dev_tnews_f1) / 3
            print(epoch, 'th epoch dev average f1 is:', dev_avg_f1)
            print(epoch, 'th epoch dev ocnli is below:')
            print_result(dev_ocnli_gold_list, dev_ocnli_pred_list)
            print(epoch, 'th epoch dev ocemotion is below:')
            print_result(dev_ocemotion_gold_list, dev_ocemotion_pred_list)
            print(epoch, 'th epoch dev tnews is below:')
            print_result(dev_tnews_gold_list, dev_tnews_pred_list)
            
            # 将dev数据集重置
            dev_data_generator.reset()
            
            # 取较好的模型
            if dev_avg_f1 > best_dev_f1:
                best_dev_f1 = dev_avg_f1
                best_epoch = epoch
                torch.save(my_net, file_path)
            print('best epoch is:', best_epoch, '; with best f1 is:', best_dev_f1)
                
if __name__ == '__main__':
    print('---------------------start training-----------------------')
    pretrained_model = './bert_pretrain_model'
    tokenizer_model = './bert_pretrain_model'
    train(batchSize=16, device='cuda:0', lr=0.0001, use_dtp=True, pretrained_model=pretrained_model, tokenizer_model=tokenizer_model, weighted_loss=True)
```

## inference的解析
```python
# 导入库函数
from net import Net
import json
import torch
import numpy as np
from transformers import BertModel, BertTokenizer
from utils import get_task_chinese


def test_csv_to_json():
    '''
    将csv转化为json文件
    TNEWS:{语料1_id：{'s1':语料}，语料2_id：{'s1':语料}……}
    OCNLI:{语料1_id：{'s1':语料，'s2':语料}，语料2_id：{'s1':语料，'s2':语料}……}
    OCEMOTION:{语料1_id：{'s1':语料}，语料2_id：{'s1':语料}……}
    '''
    for e in ['TNEWS', 'OCNLI', 'OCEMOTION']:
        with open('./tianchi_datasets/' + e + '/test.csv') as fr:
            with open('./tianchi_datasets/' + e + '/test.json', 'w') as fw:
                json_dict = dict()
                for line in fr:
                    tmp_list = line.strip().split('\t')
                    json_dict[tmp_list[0]] = dict()
                    json_dict[tmp_list[0]]['s1'] = tmp_list[1]
                    if e == 'OCNLI':
                        json_dict[tmp_list[0]]['s2'] = tmp_list[2]
                fw.write(json.dumps(json_dict))
                
def inference_warpper(tokenizer_model):
    '''
    预测
    '''
    # 打开数据
    ocnli_test = dict()
    with open('./tianchi_datasets/OCNLI/test.json') as f:
        for line in f:
            ocnli_test = json.loads(line)
            break
        
    ocemotion_test = dict()
    with open('./tianchi_datasets/OCEMOTION/test.json') as f:
        for line in f:
            ocemotion_test = json.loads(line)
            break
        
    tnews_test = dict()
    with open('./tianchi_datasets/TNEWS/test.json') as f:
        for line in f:
            tnews_test = json.loads(line)
            break
    
    # 打开label，{数据集：[label]}
    label_dict = dict()
    with open('./tianchi_datasets/label.json') as f:
        for line in f:
            label_dict = json.loads(line)
            break
    # 打开最优模型
    model = torch.load('./saved_best.pt') # 打开模型
    tokenizer = BertTokenizer.from_pretrained(tokenizer_model) # 导入分词器
    inference('./submission/ocnli_predict.json', ocnli_test, model, tokenizer, label_dict['OCNLI'], 'ocnli', 'cuda:0', 64, True) # 进行预测
    inference('./submission/ocemotion_predict.json', ocemotion_test, model, tokenizer, label_dict['OCEMOTION'], 'ocemotion', 'cuda:0', 64, True) # 进行预测
    inference('./submission/tnews_predict.json', tnews_test, model, tokenizer, label_dict['TNEWS'], 'tnews', 'cuda:0', 64, True) # 进行预测
        
def inference(path, data_dict, model, tokenizer, idx2label, task_type, device='cuda:0', batchSize=64, print_result=True):
    '''
    path：生成测试后的数据路径
    data_dict：训练数据
    model：模型
    tokenizer：分词器
    idx2label：idx2label
    task_type：任务种类
    device：设备
    batchSize：每次向模型喂入多少数据
    print_result：bool,是否打印数据
    '''
    if task_type != 'ocnli' and task_type != 'ocemotion' and task_type != 'tnews':
        print('task_type is incorrect!')
        return
    model.to(device, non_blocking=True)
    model.eval()
    ids_list = [k for k, _ in data_dict.items()] # 将所有语料的ID提取出来
    next_start_ids = 0
    with torch.no_grad(): # 预测，不需要进行梯度计算，不需要进行反向传播
        with open(path, 'w') as f:
            while next_start_ids < len(ids_list):
                cur_ids_list = ids_list[next_start_ids: next_start_ids + batchSize] # 取出batch_size大小的数据的id
                next_start_ids += batchSize # 更新next_start_ids
                # 分词
                if task_type == 'ocnli':
                    flower = tokenizer([data_dict[idx]['s1'] for idx in cur_ids_list], [data_dict[idx]['s2'] for idx in cur_ids_list], add_special_tokens=True, padding=True, return_tensors='pt') # ocnli有s1也有s2
                else:
                    flower = tokenizer([data_dict[idx]['s1'] for idx in cur_ids_list], add_special_tokens=True, padding=True, return_tensors='pt')
                # 将分词结果放入设备中
                input_ids = flower['input_ids'].to(device, non_blocking=True)
                token_type_ids = flower['token_type_ids'].to(device, non_blocking=True)
                attention_mask = flower['attention_mask'].to(device, non_blocking=True)
                # 初始化ocnli_ids，ocemotion_ids，tnews_ids
                ocnli_ids = torch.tensor([]).to(device, non_blocking=True)
                ocemotion_ids = torch.tensor([]).to(device, non_blocking=True)
                tnews_ids = torch.tensor([]).to(device, non_blocking=True)
                # 针对不同的任务，填充相应的XXX_ids，其他XXX_ids不变
                if task_type == 'ocnli':
                    ocnli_ids = torch.tensor([i for i in range(len(cur_ids_list))]).to(device, non_blocking=True)
                elif task_type == 'ocemotion':
                    ocemotion_ids = torch.tensor([i for i in range(len(cur_ids_list))]).to(device, non_blocking=True)
                else:
                    tnews_ids = torch.tensor([i for i in range(len(cur_ids_list))]).to(device, non_blocking=True)
                # 完成预测
                ocnli_out, ocemotion_out, tnews_out = model(input_ids, ocnli_ids, ocemotion_ids, tnews_ids, token_type_ids, attention_mask)
                # 完成从pred到label的预测
                if task_type == 'ocnli':
                    pred = torch.argmax(ocnli_out, axis=1)
                elif task_type == 'ocemotion':
                    pred = torch.argmax(ocemotion_out, axis=1)
                else:
                    pred = torch.argmax(tnews_out, axis=1)
                # 将index转化为label的转化
                pred_final = [idx2label[e] for e in np.array(pred.cpu()).tolist()]
                #torch.cuda.empty_cache()
                for i, idx in enumerate(cur_ids_list): # 依次取出当前预测句子的idx
                    # 打印结果
                    if print_result:
                        print_str = '[ ' + task_type + ' : ' + 'sentence one: ' + data_dict[idx]['s1']
                        if task_type == 'ocnli':
                            print_str += '; sentence two: ' + data_dict[idx]['s2']
                        print_str += '; result: ' + pred_final[i] + ' ]'
                        print(print_str)
                    # 保存结果{id:label}
                    single_result_dict = dict()
                    single_result_dict['id'] = idx
                    single_result_dict['label'] = pred_final[i]
                    f.write(json.dumps(single_result_dict, ensure_ascii=False))
                    # 最后一个语料不需要换行
                    if not (next_start_ids >= len(ids_list) and i == len(cur_ids_list) - 1): 
                        # next_start_ids >= len(ids_list) 最后一轮batch; i == len(cur_ids_list) - 1 一个batch最后一个语料
                        f.write('\n')
                        
if __name__ == '__main__':
    test_csv_to_json()
    print('---------------------------------start inference-----------------------------')
    inference_warpper(tokenizer_model='./bert_pretrain_model')
```
