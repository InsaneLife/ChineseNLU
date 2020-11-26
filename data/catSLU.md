# map域
用于分类的指标，建立分类的category

用于分类的指标：
## request
'request-剩余路程', 'request-位置', 'request-剩余距离', 'request-前方路况', 'request-路况', 'request-剩余时间' 

## inform
'inform-操作-退出','inform-value-dontcare','inform-请求类型-周边','inform-请求类型-最近','inform-操作-重新导航','inform-操作-继续导航','inform-终点名称-公司'


## 注意
'deny-终点名称': 2, 'inform-出行方式': 1
只出现在测试集中。

inform 操作 退出
inform 操作 继续导航

# music
## request
request-歌曲名

## inform
'inform-value-dontcare', 'inform-歌曲名-dontcare'

{'inform-操作': 1809, 'inform-对象': 802, 'inform-歌曲名': 1370, 'inform-歌曲数量': 475, 'inform-音乐类型': 176, 'inform-页码': 1, 'inform-歌手名': 576, 'inform-语种': 17, 'inform-乐器': 5, 'inform-value': 7, 'deny-操作': 10, 'inform-主题': 16, 'inform-主题曲类型': 8, 'inform-序列号': 13, 'inform-专辑名': 11, 'inform-适用人群': 5, 'inform-适用人名': 6, 'inform-音乐场景': 1, 'inform-应用名': 1, 'inform-适用年龄': 1, 'inform-音乐风格': 1, 'deny-歌手名': 1, 'deny-对象': 2}

{'inform-value-dontcare': 7, 'inform-歌曲名-dontcare': 6}

# video
## request
findalts-片名


## inform
都在query中


# weather
## request
{'request-天气': 1596, 'request-空气质量': 5, 'request-温度': 106, 'request-最低温度': 17, 'request-城市': 2, 'request-湿度': 3, 'request-穿衣指数': 5, 'request-PM2.5': 7, 'request-具体时间': 5, 'request-最高温度': 1}
## inform
"inform-气象-日落" 只在测试集中出现


{'inform-城市': 669, 'inform-日期': 1565, 'confirm-气象': 166, 'confirm-温度': 36, 'inform-区域': 67, 'confirm-装备': 27, 'inform-时间': 61, 'inform-气象': 7, 'inform-省份': 22, 'inform-场景': 8, 'confirm-服装': 3, 'inform-国家': 5, 'inform-地点': 4, 'confirm-活动': 6, 'confirm-湿度': 1, 'confirm-空气质量': 1, 'inform-风景点': 1, 'deny-城市': 1}