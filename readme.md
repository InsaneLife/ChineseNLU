
中文对话语义理解（单轮）

# 数据集
## [CSTSLU](https://dl.acm.org/doi/10.1145/3340555.3356098)

之前的一些对话数据集集中于语义理解，而工业界真实情况ASR也会有错误，往往被忽略。[CSTSLU](https://dl.acm.org/doi/10.1145/3340555.3356098)而是一个中文语音+NLU文本理解的对话数据集，可以从语音信号到理解端到端进行实验，例如直接从音素建模语言理解（而非word or token）。

数据统计：

![image-20200910233858454](pic/image-20200910233858454.png)

官方说明手册：[CATSLU](https://sites.google.com/view/catslu/handbook)
数据下载：[https://sites.google.com/view/CATSLU/home](https://sites.google.com/view/CATSLU/home)

## SMP

这是一系类数据集，每年都会有新的数据集放出。


### SMP-2019-NLU

包含领域分类、意图识别和语义槽填充三项子任务的数据集。训练数据集下载：[trian.json](./dialogue/SMP-2019-NLU/train.json)，目前只获取到训练集，如果有同学有测试集，欢迎提供。

|         | Train |
| ------- | ----- |
| Domain  | 24    |
| Intent  | 29    |
| Slot    | 63    |
| Samples | 2579  |



### SMP-2017

中文对话意图识别数据集，官方git和数据: [https://github.com/HITlilingzhi/SMP2017ECDT-DATA](https://github.com/HITlilingzhi/SMP2017ECDT-DATA)

数据集：

|               | Train |
| ------------- | ----- |
| Train samples | 2299  |
| Dev samples   | 770   |
| Test samples  | 666   |
| Domain        | 31    |

论文：[https://arxiv.org/abs/1709.10217  ](