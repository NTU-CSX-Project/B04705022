
# coding: utf-8

# ## import

# In[757]:


import pandas as pd
import numpy as np
import json
import snownlp
from snownlp import SnowNLP
from sklearn import preprocessing
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Dropout
from numpy import linalg as LA


# ## import data

# In[13]:


data= pd.read_csv("ted_main.csv")
data.head()


# In[282]:


EDA_data=data[['title','description','event','duration','languages','tags','ratings','views']]
EDA_data


# ## get important event

# In[670]:


impute_grps = EDA_data.pivot_table(values=["views"], index=["event"], aggfunc=np.mean)
print(impute_grps)
top_ten_event=impute_grps.views.nlargest(10).index
EDA_data.loc[~EDA_data.event.isin(top_ten_event),'event']='others'
EDA_data.event


# In[738]:


one_hot_event=pd.get_dummies(EDA_data.event)
one_hot_event
#pd.concat([EDA_data, one_hot_event], axis=1)


# ## get title/describetion sentiment 

# In[625]:


def to_sentiment(description):
    s=SnowNLP(description)
    return(s.sentiments)
description_sentiment=[]
nonlinear_description_sentiment=[]
title_sentiment=[]
nonlinear_title_sentiment=[]
for description in EDA_data.description:
    description_sentiment.append(to_sentiment(description))
    nonlinear_description_sentiment.append(1/to_sentiment(description))
for title in EDA_data.title:
    title_sentiment.append(to_sentiment(title))
    nonlinear_title_sentiment.append(1/to_sentiment(title))
description_sentiment=pd.Series(description_sentiment)
nonlinear_description_sentiment=pd.Series(nonlinear_description_sentiment)
title_sentiment=pd.Series(title_sentiment)
nonlinear_title_sentiment=pd.Series(nonlinear_title_sentiment)
print(nonlinear_title_sentiment.max())


# ## clean up tags

# In[760]:


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def map_new(x,true_tags):
    max_match=0
    max_index = 0
    for i in true_tags:
        if similar(i,x)>max_match:
            max_match = similar(i,x)
            max_index = i
    print(max_index)
    return(max_index)

def get_flat_tags(df):
    tags=[]
    for tag in df.tags:
        old=tag[1:len(df.tags[1])-1]
        nospace=old.replace(" ", "")
        clean=nospace.replace("'", "")
        clean = clean.replace("]","")
        tags.append(clean.split(","))
    df['clean_tags']=tags
    flat_tags = sum(tags,[])
    return(flat_tags)

def get_unique_tags(flat_tags):
    unique_tags = pd.Series(flat_tags).unique()
    unique_tags = sorted(unique_tags)
    return(unique_tags)

def get_true_tags(unique_tags):
    tag_num = len(unique_tags)
    true_tags = []
    for i in range(0,tag_num-1):
        if similar(unique_tags[i],unique_tags[i+1])<0.6:
            true_tags.append(unique_tags[i+1])
    return(true_tags)
            
def append_new_tags(df,true_tags):
    new_tags = []
    for tagings in df.clean_tags:
        new_tag = []
        for tag in tagings:
            new_tag.append(map_new(tag,true_tags))
        new_tags.append(new_tag)
    return(new_tags)


# In[306]:


unique_tags = get_unique_tags(EDA_data)


# In[315]:


true_tags = get_true_tags(unique_tags)
true_tags
len(true_tags)


# In[759]:


new_tags = append_new_tags(EDA_data,true_tags)


# ## get important tags

# In[776]:


from collections import Counter
flat_tags = sum(new_tags,[])
count_tags=Counter(flat_tags)
Counter_true_tags = count_tags.most_common(50)
ten_most = list(map(lambda x:(x[0]),Counter_true_tags))

def find_popular(x):
    temp = []
    for i in x:
        if i not in ten_most:
            temp.append('others')
        else:
            temp.append(i)
    return temp
#     EDA_data.loc[~x.isin(ten_most),'clean_tags']='others'
#     x = list(x)

main_tags=list(map(find_popular,list(new_tags)))
Counter_true_tags


# ## one hot encode tags

# In[247]:


def to_onehot(tags):
    sum_tags = sum(tags,[])
    unique_tags = list(set(sum_tags))
    sum_tags
    # 2. LabelEncoder
    le = preprocessing.LabelEncoder()
    
    le.fit(sum_tags)
    print("Label fit all tags")
    sum_tags_2 = le.fit_transform(sum_tags)
    print("all tag Label",sum_tags_2.shape)
    # print(list(le.classes_))
    encode_tags=[]
    i=0
    for x in tags :
        x=(le.transform(x))
        encode_tags.append(x)
        i=i+1
        if i%100==0:
            print("encode num ",i," done")
    # 2. onehot
    enc = preprocessing.OneHotEncoder()
    print("onehot fit all tags")
    enc.fit(sum_tags_2.reshape(17161,1))
    onehotlabels = enc.transform(sum_tags_2.reshape(17161,1)).toarray()
    print("all tag one hot",onehotlabels.shape)
    one_hot_tags=[]
    i=0
    for x in encode_tags:
        onehotlabels = enc.transform(x.reshape(-1,1)).toarray()
        one_hot_tags.append(onehotlabels)
        i=i+1
        if i%100==0:
            print("encode num ",i," done\nresults:",onehotlabels)
    return(one_hot_tags)


# In[280]:


def sumtags(one_hot_tags):
    one_hot_sum_tags=[]
    for aTadTalk in one_hot_tags:
        x=0
        for tag in aTadTalk:
            x=x+tag
#         print(x.sum())
        one_hot_sum_tags.append(x)
    return (one_hot_sum_tags)


# In[777]:


one_hot_tags=to_onehot(main_tags)
one_hot_sum_tags=sumtags(one_hot_tags)
one_hot_sum_tags=pd.DataFrame(np.array(one_hot_sum_tags))


# ## normalize

# In[710]:


scale_duration=pd.Series(preprocessing.scale(np.array(EDA_data.duration)))
scale_languages=pd.Series(preprocessing.scale(np.array(EDA_data.languages)))


# ## get xs together

# In[784]:


EDA_done=pd.concat([nonlinear_title_sentiment,nonlinear_description_sentiment,scale_duration,one_hot_event,one_hot_sum_tags], axis=1)
EDA_done.shape


# ## preprocess ys

# In[658]:


EDA_data['q']=0
EDA_data.loc[EDA_data.views>EDA_data.views.quantile(0.25),'q']=1
EDA_data.loc[EDA_data.views>EDA_data.views.quantile(0.5),'q']=2
EDA_data.loc[EDA_data.views>EDA_data.views.quantile(0.75),'q']=3
EDA_data.head()


# In[704]:


scale_views=pd.Series(preprocessing.scale(np.array(EDA_data['views'])))


# ## combine xs、ys

# In[785]:


EDA_done_withY=pd.concat([EDA_done,one_hot_q],axis=1)


# In[742]:


def get_train_test(csv):
    msk = np.random.rand(len(csv))<0.8
    train = csv[msk]
    test = csv[~msk]
    return(train,test)
def get_x_y(train,xs_count,ys_count):
    x_train = np.array(train.iloc[0:, 0:xs_count])
    y_train = np.array(train.iloc[0:,xs_count:xs_count+ys_count]).reshape(len(train),ys_count)
    return(x_train,y_train)


# In[786]:


XS_COUNT=65
YS_COUNT=4
train,test=get_train_test(EDA_done_withY)
# train.loc[:,]
x_train,y_train=get_x_y(train,XS_COUNT,YS_COUNT)
x_test,y_test=get_x_y(test,XS_COUNT,YS_COUNT)
y_train


# In[793]:


model = Sequential()
# model.add(Dense(units=100,
#                input_dim=13,
#                kernel_initializer='uniform',
#                activation='relu'))
model.add(Dense(units=50,
                input_dim=XS_COUNT,
               kernel_initializer='uniform',
               activation='tanh'))
model.add(Dropout(rate=0.15))
model.add(Dense(units=YS_COUNT,
               kernel_initializer='uniform',
               activation='softmax'))

model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])



# In[794]:


train_history = model.fit(x=x_train, 
                          y=y_train,
                          validation_split=0.05,
                          epochs=100, 
                          batch_size=30,
                          verbose=1)


# In[795]:


scores = model.evaluate(x_test,y_test)
scores


# In[640]:


get_ipython().system('jupyter nbconvert --to script Untitled.ipynb')

