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
    
    for e in ['TNEWS', 'OCNLI', 'OCEMOTION']:
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
    
    for e in ['TNEWS', 'OCNLI', 'OCEMOTION']:
        for name in ['dev', 'train']:
            cur_path = './tianchi_datasets/' + e + '/' + name + '.json'
            data_name = e + '_' + name
            print_one_data(cur_path, data_name)
            
    print_one_data('./tianchi_datasets/label.json', 'label_set')
    
if __name__ == '__main__':
    print('-------------------------------start-----------------------------------')
    split_dataset(dev_data_cnt=3000)
    generate_data()
    print('-------------------------------finish-----------------------------------')
```


## train的解析

## inference的解析
