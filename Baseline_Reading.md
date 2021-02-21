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

## train的解析

## inference的解析
