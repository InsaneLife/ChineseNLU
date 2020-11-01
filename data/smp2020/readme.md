# Few-joint Dataset

The Files are dataset for **FewJoint: A Few-shot Learning Benchmark for Joint Language Understanding**

#### Content:

##### SMP2020-ECDT_Origin_3shot
The contest version for [SMP2020-ECDT track1](https://smp2020.aconf.cn/smp.html#3)

##### FewJoint_Xshot
The refined version for X-shot language understanding.
(Stricter shot limits with both human and machine checks.)


#### Format:
File structure:
```
FewJoint_Xshot
|-- dev: data of dev domains
    |--support: support sets, number in file name is domain id
    |--test: query sets, number in file name is domain id
    |--correct: query sets with correct results, number in file name is domain id
    |--predict: files use to fill model prediction for evaluation
|-- test: data of test domains
    |--support: support sets, number in file name is domain id
    |--test: query sets, number in file name is domain id
    |--correct: query sets with correct results, number in file name is domain id
|-- train: data of train domains
    |-- source.json: all training data
```

Json data format for each instance:
```
[
   {
      "text": "查询今天的天气",
      "domain": "查询助手",
      "intent": "查天气",
      "slots"：{
       "日期": "今天"
     }
  },
]
```
