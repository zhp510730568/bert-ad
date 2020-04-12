本项目用于多卡训练的支持。
# 介绍
　　目前，开源bert只支持单显卡训练，无法利用多显卡实现数据并行训练，当然，可以使用分布式方式实现并行训练，但是分布式方式管理比较麻烦，进程和数据需要手动管理。本改进主要针对单机多卡的机器，充分利用多卡实现加快训练和提高batch size大小的目的，使用方法和正常使用bert训练几乎没有区别，只比bert多加了个参数，该参数指定了显卡的数量。
    改进使用between in-graph replication方式。训练时，每个gpu负责训练一个或者多个子图，该子图有bert的完整逻辑，多块显卡训练完成后会聚合每个子图的参数，然后计算参数的平均值，最后，修改参数。除此之外，源码加了测试集的预测逻辑，会在checkpoint目录创建predict.txt文件，每行是测试集每条样本的预测结果，预测和评估方式使用CUDA_VISIBLE_DEVICES指定的第一块显卡，这部分和原bert没有区别。新加参数do_predict指定是否输出测试集预测结果，n_gpus指定使用几块显卡，可以实现一卡对应多个子图，实际训练batch_size是train_batch_size * n_gpus，启动脚本如下所示，红色部分是改动部分。


>export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6

>python -u `run_classifier_mgpu.py` --task_name=high_level  --do_train=true --do_predict=false --do_eval=false  
>--data_dir=./data/high_level/   --vocab_file=./model/chinese_L-12_H-768_A-12/vocab.txt   --bert_config_file=./model/chinese_L-12_H-768_A-12/bert_config.json  
>--init_checkpoint=./model/chinese_L-12_H-768_A-12/bert_model.ckpt   --max_seq_length=512 --train_batch_size=12 
>--learning_rate=2e-5 --num_train_epochs=5 --output_dir=./model/highlevel/model_mgpu_test/ --warmup_proportion=0.1 
>--eval_batch_size=1 --n_gpus=6
