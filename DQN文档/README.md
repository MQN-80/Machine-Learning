# 1.环境配置
## 1.1 tensorflow版本
* tensorflow>2.2.0 只要tensor2均可
* tensorlayer>2.2.3
* tensorflow-probablity:0.6.0
## 1.2 gym环境
* pip install gym[all]     安装所有gym环境
## 2. 程序执行
* 程序可采用命令行执行，具体参数含义可见程序
命令行执行训练模式，可自行输入训练次数
* python DQN.py --train=True --train_episodes=1000  
命令行加载预训练模型,文件中已经有训练好的模型，自己也可以训练
* python  DQN.py --test=True --test_episodes=50

