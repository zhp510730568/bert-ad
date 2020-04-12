# bert-ad
本项目用于多卡训练的支持。
## 介绍
　　目前，开源bert只支持单显卡训练，无法利用多显卡实现数据并行训练，当然，可以使用分布式方式实现并行训练，但是分布式方式管理比较麻烦，进程和数据需要手动管理。本改进主要针对单机多卡的机器，充分利用多卡实现加快训练和提高batch size大小的目的，使用方法和正常使用bert训练几乎没有区别，只比bert多加了个参数，该参数指定了显卡的数量。
    改进使用between in-graph replication方式。训练时，每个gpu负责训练一个或者多个子图，该子图有bert的完整逻辑，多块显卡训练完成后会聚合每个子图的参数，然后计算参数的平均值，最后，修改参数。除此之外，源码加了测试集的预测逻辑，会在checkpoint目录创建predict.txt文件，每行是测试集每条样本的预测结果，预测和评估方式使用CUDA_VISIBLE_DEVICES指定的第一块显卡，这部分和原bert没有区别。新加参数do_predict指定是否输出测试集预测结果，n_gpus指定使用几块显卡，可以实现一卡对应多个子图，实际训练batch_size是train_batch_size * n_gpus，启动脚本如下所示，红色部分是改动部分。多卡预训练同样支持，类似使用方法，项目还包括文本改写的预训练以及微调的代码，还包括训练数据预处理，有问题可以提issue咨询。


>export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6

>python -u `run_classifier_mgpu.py` --task_name=high_level  --do_train=true --do_predict=false --do_eval=false  
>--data_dir=./data/high_level/   --vocab_file=./model/chinese_L-12_H-768_A-12/vocab.txt   --bert_config_file=./model/chinese_L-12_H-768_A-12/bert_config.json  
>--init_checkpoint=./model/chinese_L-12_H-768_A-12/bert_model.ckpt   --max_seq_length=512 --train_batch_size=12 
>--learning_rate=2e-5 --num_train_epochs=5 --output_dir=./model/highlevel/model_mgpu_test/ --warmup_proportion=0.1 
>--eval_batch_size=1 --n_gpus=6

## 测试
表1是改进bert的训练速度，原始bert单卡对应长度速度分别是100examples/sec、30examples/sec、16examples/sec和8examples/sec，速度随显卡数增多成线性增长，相同num_train_epochs训练速度大幅缩短。即使改进bert使用单卡也要比原bert要提升百分之十左右，像p100显卡，显存16G，对max_seq_length较短的训练数据使用改进bert配置两个子图能够明显提升训练速度，显卡利用率要显著提高。多次测试对比，改进bert参数n_gpus设置为2和原bert模型训练准确率相比几乎相同，提高n_gpus为6，准确率有零点几的百分比下降。

![](https://github.com/zhp510730568/bert-ad/blob/master/%E9%80%9F%E5%BA%A6%E5%AF%B9%E6%AF%94.png)
## 建议
不要太过迷恋开源预训练模型，现在很多预训练模型大同小异，无非是更多的数据加上不同的预训练任务，其实领域内数据对下游任务要重要的多，仔细看下各个开源预训练模型所用数据就会发现，这些数据和自己业务数据差异较大，可能达不到最好的fine-tuning效果，建议方法用自己数据在开源预训练基础上再预训练，这种方法对小公司来说这更重要，一两千万数据预训练上百万个batch就能看出明显提升。举个个人经验，预训练数据有个游戏名"赤痕"只出现了三次，标题分类只有预训练后的能正确分类，各种开源都哑炮。过于强大的预训练模型使用也有风险，可能会过拟合，不是说会过拟合训练数据，比如资讯类，随时间推移使用模型的数据分布可能会发生变化，导致实际性能衰减厉害。下图左图是评估数据在一年前标注数据训练过程中的准确率变化曲线，右图是训练数据测试集在训练过程中准确率变化曲线，自己分析数据会发现里面很多最近的热门事件预测准确率只有50%左右，反而使用简单模型效果会更稳定。
![](https://github.com/zhp510730568/bert-ad/blob/master/%E8%BF%87%E6%8B%9F%E5%90%88.png)
## 改进
这种方式显然不适合GPU量很大的模型训练，测试机器都是单机多卡，GPU卡数量不多，同步代价比较低。这种方式显然不适合需要使用大量GPU进行加速的训练，这就需要使用分布式方式进行训练，但tensorflow提供的分布式方案存在GPU利用率低下的问题，有公司测试结果显示GPU利用率在50%左右，百度提出了新的梯度更新方法，使用不同的算法来平均梯度，并让这些梯度在所有节点之间交流，这被称为 ring-allreduce，并开源了tensorflow实现，github地址如下列表。除此之外，Uber也开源tensorflow的实现，开源项目是Horovod，相比ring-allreduce做了一些性能优化，并大幅提高易用性。

ring-allreduce blog: http://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/

ring-allreduce github: https://github.com/baidu-research/tensorflow-allreduce

Horovod github: https://github.com/horovod/horovod
