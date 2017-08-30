#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 16:58:14 2017

@author: Sandra
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation, metrics

class flight:
    
    def load(self):
        X = []
        y = []
        reader = pd.read_csv('../Data/train/flight_information.csv',iterator=True,encoding="gbk")
        loop = True
        chunkSize = 100000
        chunks = []
        while loop:
            try:
                chunk = reader.get_chunk(chunkSize)
                chunks.append(self.addDelay(chunk, X, y))
                #break
            except StopIteration:
                loop = False
                print('------->load data success')
        df = pd.concat(chunks, ignore_index=True)
        
        print('样本数量: %d'%len(df))
        print('负样本数量: %d'%len(df[df['isMoreThan3'] == 1]))
        
        return [X,y]
    
    def addDelay(self, df, X, y):
        df = df.dropna()
        df['delay'] = df[u'实际起飞时间'] - df[u'计划起飞时间']
        df['isMoreThan3'] = df['delay'].apply(lambda x: x>3*3600 and 1 or 0)
        for index,row in df.iterrows():
            #row[u'出发机场'],row[u'到达机场'],row[u'航班编号'],
            X.append([row[u'计划起飞时间'],row[u'计划到达时间'],row[u'飞机编号']])
            y.append(row['isMoreThan3'])
        print '------->load 100000'
        return df

def randomForest(data):
    X = data[0]
    y = data[1]

    print('------->prepare')
    
    rf = RandomForestClassifier(oob_score=True, random_state=10)
    rf.fit(X,y)
                  
    print(rf.oob_score_)
    y_predprob = rf.predict_proba(X)[:,1]
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))

if __name__ == '__main__':
    
    f = flight()
    data = f.load()
    randomForest(data)
