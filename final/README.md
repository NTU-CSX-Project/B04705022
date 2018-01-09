#Ted Talks Analysis
[PPT url](https://docs.google.com/presentation/d/1uk5HWMCUyM99X_mHZfGgpKgFZo6H4NtKEhA0RYWVkZ8/edit#slide=id.g2dee5522c0_0_134
) 
## 資料集
- description
- tags
- title
- event
- ⇒ views
## 前處理
1. description , title
使用snoeNLP取得情緒分析
並正規化
2.ags , event  
=> one-hot encoding
找出常出現，average views 高的 tags
## model
using keras
- DNN
- Dropout 
- regularization



