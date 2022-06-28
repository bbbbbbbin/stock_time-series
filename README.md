# stock_time-series
2019年毕业设计----基于时间序列的股票投资方案

## 论文摘要

想要从股票中获取最大的利益，则要对于股票进行分析及预测并制定一定的投资方案，依据股票的历史价格建立适当的模型是一种可行的方法。本文主要阐述基于 HP 滤波以及 ARMA 的模型方法。选取了来自 DC 竞赛网站中关于量化投资竞赛中roundA 的 111 支不同股票的 488 天收盘价的数据。使用 python 语言，对数据进行 HP滤波，分离出两个时间序列，分别进行建模，并对模型进行检验及预测，最后筛选出收益较好的股票。

## 论文关键词

**时间序列，ARMA模型，HP滤波**

## 主要代码

read_data.ipynb为读取和处理数据的代码

func_test.ipynb为数据模型探索代码

applica.py 为最终应用数据测试得结论的代码
