
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
from os.path import join
import os
from sklearn import preprocessing
from sklearn import tree
import pickle
def find_missing_cols(input_data):
    return [col for col in input_data.columns if input_data[col].isnull().any()]
def find_non_numeric_cols(input_data):
    return [col for col in input_data.columns if input_data[col].dtype=='object']
def find_int64_cols(input_data):
    return [col for col in input_data.columns if input_data[col].dtype=='int64']
def find_float64_cols(input_data):
    return [col for col in input_data.columns if input_data[col].dtype=='float64']
def get_label_encoder(LE_path=join('model','LE')):
    LE_table={}
    LE_lst=os.listdir(LE_path)
    for file in LE_lst:
        with open(join(LE_path,file),'rb') as f: 
            LE_table[file]=pickle.load(f)
    return LE_table
def fill_nan(data):
    LE_table=get_label_encoder()
    ori_missing_cols=find_missing_cols(data)
    
    new_data = data.copy()
    new_data = new_data.drop(ori_missing_cols, axis=1)
    new_non_numeric_cols=find_non_numeric_cols(new_data)
    if len(LE_table)>0:
        for col in new_non_numeric_cols:
            new_data.at[:,col]=LE_table[col].transform(new_data[col])
    else:
        for col in new_non_numeric_cols:
            le = preprocessing.LabelEncoder()
            le.fit(new_data[col])
            new_data.at[:,col]=le.transform(new_data[col])
            LE_table[col]=le
    new_int64_cols = find_int64_cols(new_data)
    new_float64_cols = find_float64_cols(new_data)
    for col in ori_missing_cols:
        if data[col].dtype=='object':
            null_rows=data[col].isnull()
            exist_rows=[i for i in null_rows.index if not null_rows[i]]
            null_rows=[i for i in null_rows.index if null_rows[i]]
            
            train_data=new_data.copy().drop(null_rows,axis=0)  #train_data
            
            train_answer=data[col].copy().drop(null_rows,axis=0)
            if col in LE_table:
                train_answer=LE_table[col].transform(train_answer)
            else:
                le = preprocessing.LabelEncoder()
                le.fit(train_answer)
                train_answer=le.transform(train_answer)               #train_answer
                LE_table[col]=le
            
            clf = tree.DecisionTreeClassifier()
            clf = clf.fit(train_data, train_answer)
            
            predict_data=new_data.copy().drop(exist_rows,axis=0)
            predict_answer=clf.predict(predict_data)
            
            train_data[col]=np.array(train_answer)
            predict_data[col]=np.array(predict_answer)
            frames=[train_data,predict_data]
            new_data=pd.concat(frames)
            
        elif data[col].dtype=='int64':
            null_rows=data[col].isnull()
            exist_rows=[i for i in null_rows.index if not null_rows[i]]
            null_rows=[i for i in null_rows.index if null_rows[i]]
            train_data=new_data.copy().drop(null_rows,axis=0)  #train_data
            train_answer=data[col].copy().drop(null_rows,axis=0)
            clf = tree.DecisionTreeRegressor()
            clf = clf.fit(train_data, train_answer)
            predict_data=new_data.copy().drop(exist_rows,axis=0)
            predict_answer=clf.predict(predict_data)
            train_data[col]=np.array(train_answer.astype(np.int64))
            predict_data[col]=np.array(predict_answer.astype(np.int64))
            frames=[train_data,predict_data]
            new_data=pd.concat(frames)
    
        elif data[col].dtype=='float64':
            null_rows=data[col].isnull()
            exist_rows=[i for i in null_rows.index if not null_rows[i]]
            null_rows=[i for i in null_rows.index if null_rows[i]]
            train_data=new_data.copy().drop(null_rows,axis=0)  #train_data
            train_answer=data[col].copy().drop(null_rows,axis=0)
            clf = tree.DecisionTreeRegressor()
            clf = clf.fit(train_data, train_answer)
            predict_data=new_data.copy().drop(exist_rows,axis=0)
            predict_answer=clf.predict(predict_data)
            train_data[col]=np.array(train_answer.astype(np.float64))
            predict_data[col]=np.array(predict_answer.astype(np.float64))
            frames=[train_data,predict_data]
            new_data=pd.concat(frames)
    return new_data,LE_table
if __name__=="__main__":
    path=join("data","train.csv")
    data=pd.read_csv(path,index_col=0)
    new_data,label_encoder_table=fill_nan(data)

    new_data.to_csv(join('data','fillnan_train.csv'))
    import pickle
    for k,v in label_encoder_table.items():
        with open(join('model',join('LE',k)),'wb') as f:
            pickle.dump(label_encoder_table[k], f)


# In[ ]:




