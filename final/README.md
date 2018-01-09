#Ted Talks Analysis
[PPT url](https://docs.google.com/presentation/d/e/2PACX-1vS8W2O1llKKdioQtsW7UeWFdfvSrOZ2kxGubmIItj-3YwJN7wuQQb3BWdxYQYVRwFb_NRRM4nAiBlKJ/pub?start=false&loop=false&delayms=3000) 
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



