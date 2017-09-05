# coding=utf-8

from __future__ import print_function
import pandas as pd
import numpy as np
import datetime, time
import json
from gensim.models import word2vec

# 模型
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer
from sklearn import svm
from sklearn.linear_model import LogisticRegression

# 评价指标  
from sklearn import metrics

# 画图
# import matplotlib.pyplot as plt

# 交叉验证 10折
from sklearn.cross_validation import cross_val_score, KFold


def load_airport_city():
    # 将机场城市表转为字典
    airport_city = {}
    df = pd.read_excel('../Data/train/airport_city.xlsx')
    for index, row in df.iterrows():
        airport_city[row[u'城市名称'].encode('utf-8')] = row[u'机场编码'].encode('utf-8')
    '''
    fl=open('../Data/train/airport_city.json', 'w')
    fl.write(json.dumps(airport_city))
    fl.close()
    '''
    print('------>Load Airport City Dict Success!')
    return airport_city

def load_test_airport_city():
    # 将机场城市表转为字典
    airport_city = {}
    df = pd.read_excel('../Data/test A/airport_city.xlsx')
    for index, row in df.iterrows():
        airport_city[row[u'城市名称']] = row[u'机场编码']
    '''
    fl=open('../Data/test A/airport_city.json', 'w')
    fl.write(json.dumps(airport_city))
    fl.close()
    '''
    print('------>Load Airport City Dict Success!')
    return airport_city

def load_weather():
    # 获取天气数据
    df = pd.read_csv('../Data/train/weather.csv')
    del df['Unnamed: 5']
    airport_city_dict = load_airport_city()
    print(airport_city_dict)

    # 在天气数据中，保留城市机场表中的城市
    df = df[df['城市'].isin(airport_city_dict.keys())]

    # 替换城市为机场
    df = df.replace(airport_city_dict)

    # 转换天气为向量
    # 天气语料
    words = ''
    for x in list(set(list(df['天气']))):
        words += x + '\n'

    # 语料写入文件
    # fsock = open("../Data/train/output/weather_word.txt", "a")
    # fsock.write(words)
    # fsock.close()

    # 读取语料
    sentences = word2vec.LineSentence("../Data/train/output/weather_word.txt")
    model = word2vec.Word2Vec(sentences, min_count=1, size=3)

    # 存储模型
    '''
    model.save('../Data/train/output/weather_word_model.txt')  
    model = word2vec.Word2Vec.load('../Data/train/output/weather_word_model.txt') 
    '''

    # 添加天气向量    
    df['weatherVec'] = df['天气'].apply(lambda x: list(model[unicode(x, "utf-8")]))
    print(df.head())
    df.to_csv('../Data/train/output/weather_airport_vec.csv', index=False)

def load_test_weather():
    # 获取天气数据
    df = pd.read_excel('../Data/test A/weather.xlsx')
    df = df.dropna()
    
    airport_city_dict = load_test_airport_city()

    # 在天气数据中，保留城市机场表中的城市
    df = df[df[u'城市'].isin(airport_city_dict.keys())]

    # 替换城市为机场
    df = df.replace(airport_city_dict)

    # 转换天气为向量
    # 天气语料
    words = ''
    for x in list(set(list(df[u'天气']))):
        words += x + '\n'

    # 语料写入文件

    fsock = open("../Data/test A/output/weather_word.txt", "a")
    fsock.write(words.encode('utf-8'))
    fsock.close()


    # 读取语料
    sentences = word2vec.LineSentence("../Data/test A/output/weather_word.txt")
    model = word2vec.Word2Vec(sentences, min_count=1, size=3)

    # 存储模型
    '''
    model.save('../Data/test A/output/weather_word_model.txt')  
    model = word2vec.Word2Vec.load('../Data/test A/output/weather_word_model.txt') 
    '''

    # 添加天气向量    
    df['weatherVec'] = df[u'天气'].apply(lambda x: list(model[x]))
    df.to_csv('../Data/test A/output/weather_airport_vec.csv', index=False,encoding='gbk')


def string2timestamp(strValue):
    try:
        d = datetime.datetime.strptime(strValue, "%Y-%m-%d %H:%M:%S")
        t = d.timetuple()
        timeStamp = int(time.mktime(t))
        timeStamp = float(str(timeStamp) + str("%06d" % d.microsecond)) / 1000000
        return timeStamp
    except ValueError as e:
        print
        e


def load_special_news():
    # 获取机场特情数据
    df = pd.read_excel('../Data/train/special_news.xlsx')

    # 缺失值
    df = df.dropna()

    # 转换为时间戳
    df[u'收集时间'] = df[u'收集时间'].apply(lambda x: string2timestamp(x[:-1]))
    df[u'开始时间'] = df[u'开始时间'].apply(lambda x: string2timestamp(x[:-1]))
    df[u'结束时间'] = df[u'结束时间'].apply(lambda x: string2timestamp(x[:-1]))

    return df

def load_test_special_news():
    # 获取机场特情数据
    df = pd.read_excel('../Data/test A/special_news.xlsx')

    # 缺失值
    df = df.dropna()

    # 转换为时间戳
    df[u'收集时间'] = df[u'收集时间'].apply(lambda x: string2timestamp(x[:-1]))
    df[u'开始时间'] = df[u'开始时间'].apply(lambda x: string2timestamp(x[:-1]))
    df[u'结束时间'] = df[u'结束时间'].apply(lambda x: string2timestamp(x[:-1]))

    return df


def load_data():

    # 获取数据
    reader = pd.read_csv('../Data/train/flight_information.csv', iterator=True, encoding="gbk")
    loop = True
    chunkSize = 100000
    chunks = []
    count = 0
    while loop:
        try:
            count += 1
            chunk = reader.get_chunk(chunkSize)
            #chunks.append(chunk)
            chunks.append(extractBasicFeature(chunk))
            print(count*chunkSize,'Extract basic feature success')
            #break
        except StopIteration:
            loop = False
            
    df = pd.concat(chunks, ignore_index=True)
    #print(len(df[u'飞机编号'].drop_duplicates())) 3410
    df.to_csv('../Feature/data_with_basic_feature.csv',index=False,encoding='gbk')
    
    df = extractFlightFeature(df)    
    df.to_csv('../Feature/data_with_flight_feature.csv',index=False,encoding='gbk')
    
    df = extractLastFlightFeature(df)
    df.to_csv('../Feature/data_with_last_flight_feature.csv',index=False,encoding='gbk')
    
    df = balanceSample(df)
    print('Balance sample success')
    
    # 出发机场天气
    weather_from = pd.read_csv('../Data/train/output/weather_airport_vec.csv')
    weather_from.columns = [u'出发机场', u'出发机场天气', u'出发机场最低气温', u'出发机场最高气温', 'date', 'weatherVecFrom']
    weather_from['weatherVecFrom'] = weather_from['weatherVecFrom'].apply(lambda x: [float(i) for i in x[1:-1].split(', ')])
    weather_from['weatherVecFrom0'] = weather_from['weatherVecFrom'].apply(lambda x: x[0])
    weather_from['weatherVecFrom1'] = weather_from['weatherVecFrom'].apply(lambda x: x[1])
    weather_from['weatherVecFrom2'] = weather_from['weatherVecFrom'].apply(lambda x: x[2])
    
    # 到达机场天气
    weather_to = pd.read_csv('../Data/train/output/weather_airport_vec.csv')
    weather_to.columns = [u'到达机场', u'到达机场天气', u'到达机场最低气温', u'到达机场最高气温', 'date', 'weatherVecTo']
    weather_to['weatherVecTo'] = weather_to['weatherVecTo'].apply(lambda x: [float(i) for i in x[1:-1].split(', ')])
    weather_to['weatherVecTo0'] = weather_to['weatherVecTo'].apply(lambda x: x[0])
    weather_to['weatherVecTo1'] = weather_to['weatherVecTo'].apply(lambda x: x[1])
    weather_to['weatherVecTo2'] = weather_to['weatherVecTo'].apply(lambda x: x[2])
    print('Load weather data success')

    # 机场特情
    sn = load_special_news()
    print('Load special news data success')

    # 添加天气特征
    data_from = pd.merge(df, weather_from, on=[u'出发机场', 'date'], how='inner')
    data = pd.merge(data_from, weather_to, on=[u'到达机场', 'date'], how='inner')
    print('Extract weather feature success')

    # 添加特情
    data['hasSpecialNews'] = 0
    for index, row in sn.iterrows():
        # print data.ix[0,u'计划起飞时间'] - row[u'结束时间']
        data[((data[u'出发机场'] == row[u'特情机场']) | (data[u'到达机场'] == row[u'特情机场'])) & (
            (data[u'计划起飞时间'] > row[u'开始时间']) & (data[u'计划起飞时间'] < row[u'结束时间']))]['hasSpecialNews'] = 1
        # break
    print('Extract special news feature success')

    # 缺失值 
    data = data.dropna()
    
    # print data[data['hasSpecialNews'] == True]
    # print data[data['timePrepareThisFlightPlan'] != 0].head()
    print('Load data with feature success')
    
    
    return data

def load_test_data():
    
    # 获取数据
    '''
    df = pd.read_csv('../Data/test A/flight_information.csv',encoding="gbk")

    df = extractBasicFeature(df)
    df = extractFlightFeature(df)        
    df = extractLastFlightFeature(df)
    
    df.to_csv('../Data/test A/output/data_feature.csv',index=False,encoding='gbk')
    '''
    df = pd.read_csv('../Data/test A/output/data_feature.csv',encoding="gbk")
    load_test_weather()

    # 出发机场天气
    weather_from = pd.read_csv('../Data/test A/output/weather_airport_vec.csv')
    weather_from.columns = [u'出发机场', u'出发机场天气', u'出发机场最低气温', u'出发机场最高气温', 'date', 'weatherVecFrom']
    weather_from['weatherVecFrom'] = weather_from['weatherVecFrom'].apply(lambda x: [float(i) for i in x[1:-1].split(', ')])
    weather_from['weatherVecFrom0'] = weather_from['weatherVecFrom'].apply(lambda x: x[0])
    weather_from['weatherVecFrom1'] = weather_from['weatherVecFrom'].apply(lambda x: x[1])
    weather_from['weatherVecFrom2'] = weather_from['weatherVecFrom'].apply(lambda x: x[2])
    
    # 到达机场天气
    weather_to = pd.read_csv('../Data/test A/output/weather_airport_vec.csv')
    weather_to.columns = [u'到达机场', u'到达机场天气', u'到达机场最低气温', u'到达机场最高气温', 'date', 'weatherVecTo']
    weather_to['weatherVecTo'] = weather_to['weatherVecTo'].apply(lambda x: [float(i) for i in x[1:-1].split(', ')])
    weather_to['weatherVecTo0'] = weather_to['weatherVecTo'].apply(lambda x: x[0])
    weather_to['weatherVecTo1'] = weather_to['weatherVecTo'].apply(lambda x: x[1])
    weather_to['weatherVecTo2'] = weather_to['weatherVecTo'].apply(lambda x: x[2])
    print('Load weather data success')

    # 机场特情
    sn = load_test_special_news()
    print('Load special news data success')

    # 添加天气特征
    data_from = pd.merge(df, weather_from, on=[u'出发机场', 'date'], how='inner')
    data = pd.merge(data_from, weather_to, on=[u'到达机场', 'date'], how='inner')
    print('Extract weather feature success')

    # 添加特情
    data['hasSpecialNews'] = 0
    for index, row in sn.iterrows():
        # print data.ix[0,u'计划起飞时间'] - row[u'结束时间']
        data[((data[u'出发机场'] == row[u'特情机场']) | (data[u'到达机场'] == row[u'特情机场'])) & (
            (data[u'计划起飞时间'] > row[u'开始时间']) & (data[u'计划起飞时间'] < row[u'结束时间']))]['hasSpecialNews'] = 1
        # break
    print('Extract special news feature success')

    # 缺失值 
    data = data.dropna()    
    
    # print data[data['hasSpecialNews'] == True]
    # print data[data['timePrepareThisFlightPlan'] != 0].head()
    print('Load test data with feature success')
    return data

def extractBasicFeature(df):

    # 缺失值
    df = df.dropna()
         

    # 延误时间
    df['delay'] = df[u'实际起飞时间'] - df[u'计划起飞时间']

    # 是否延误超过3小时 或 航班取消
    df['isMoreThan3'] = df['delay'].apply(lambda x: x >= 3 * 3600 and True) | df[u'航班是否取消'].apply(
        lambda x: x == u'取消' and True)

    # 起飞前两小时时间
    df['timeBefore2Hour'] = df[u'计划起飞时间'].apply(lambda x: x - 2 * 3600)

    # 飞行时间
    df['flightTime'] = df[u'计划到达时间'] - df[u'计划起飞时间']

    # 日期
    df[u'date'] = df[u'计划起飞时间'].apply(lambda x: datetime.datetime.utcfromtimestamp(x).strftime("%Y-%m-%d"))
    
    return df
          

def extractFlightFeature(df):
    
    # 添加本航班特征
    flight_feature = []
    for flight in df.groupby([u'航班编号', u'出发机场']):
        
        print(flight[0],' Extract flight feature success')
        
        describe = flight[1]['delay'].describe()
        if len(flight[1]) > 1:
            flight_feature.append(
                [flight[0][0], flight[0][1], describe['mean'], describe['std'], describe['max'], describe['50%']])
        else:
            flight_feature.append([flight[0][0], flight[0][1], describe['mean'], 0, describe['max'], describe['50%']])
    df_flight_feature = pd.DataFrame(flight_feature,
                                     columns=[u'航班编号', u'出发机场', u'平均延误时间', u'延误时间标准差', u'最大延误时间', u'延误时间中位数'])
    df = pd.merge(df, df_flight_feature, on=[u'航班编号', u'出发机场'], how='inner')
    return df

def extractLastFlightFeature(df):
    
    # 添加前序航班特征
    dfg_all = []
    for flight in df.groupby([u'飞机编号', u'date']):
        
        print(flight[0],' Extract last flight feature success')
        
        dfg = flight[1].sort_values(u'计划起飞时间')
        # dfg['lastFlight'] = 0
        dfg['timeLastFlightDelay'] = 0
        dfg['timePrepareThisFlightRemain'] = 0
        dfg['timePrepareThisFlightPlan'] = 0
        # dfg['timePrepareThisFlightAct'] = 0

        # 该飞机一天飞行次数超过两次
        if len(dfg) > 1:
            index = dfg.index
            for i in range(1, len(index)):
                # 前序航班编号
                dfg.ix[index[i], 'lastFlight'] = dfg.ix[index[i - 1], u'航班编号']

                # 上次飞行延误时间
                delay = dfg.ix[index[i - 1], u'delay']
                dfg.ix[index[i], 'timeLastFlightDelay'] = delay

                # 本次飞行预计起飞时间 - 上次飞行实际到达时间
                # 若本航班不延误，准备本次飞行的剩余时间
                time1 = dfg.ix[index[i], u'计划起飞时间'] - dfg.ix[index[i - 1], u'实际到达时间']
                dfg.ix[index[i], 'timePrepareThisFlightRemain'] = time1

                # 本次飞行预计起飞时间 - 上次飞行计划到达时间
                # 计划准备本次飞行的时间
                time2 = dfg.ix[index[i], u'计划起飞时间'] - dfg.ix[index[i - 1], u'计划到达时间']
                dfg.ix[index[i], 'timePrepareThisFlightPlan'] = time2

                # 本次飞行实际起飞时间 - 上次飞行实际到达时间
                # 实际准备本次飞行的时间
                # time3 = dfg.ix[index[i], u'实际起飞时间'] - dfg.ix[index[i-1], u'实际到达时间']
                # dfg.ix[index[i], 'timePrepareThisFlightAct'] = time3

        dfg_all.append(dfg)

    result = pd.concat(dfg_all, ignore_index=True)
    return result

def balanceSample(result):
    # 对正样本抽样
    # 负样本
    result_n = result[result['isMoreThan3'] == 1]
    # 正样本
    result_p = result[result['isMoreThan3'] != 1]
    # 数量
    count_n = len(result_n)
    # print '负样本数量: ',count_n
    # print '取样前正样本数量: ',len(result_p)
    # 抽样
    result_p = result_p.sample(n=2 * count_n)
    # print '取样后正样本数量: ',len(result_p)
    # 连接正样本与负样本
    data = pd.concat([result_n, result_p])

    # print('------->Balance sample success')
    return data


def load_data_with_feature():
    # 获取数据
    reader = pd.read_csv('../Feature/data_with_last_flight_feature.csv', iterator=True, encoding="gbk")
    loop = True
    chunkSize = 100000
    chunks = []
    count = 0
    while loop:
        try:
            count += 1
            chunk = reader.get_chunk(chunkSize)
            chunks.append(chunk)
            print(count*chunkSize,'Load data with feature success')
            #break
        except StopIteration:
            loop = False
            
    df = pd.concat(chunks, ignore_index=True)
    df = balanceSample(df)
    print('Balance sample success')
    
    # 出发机场天气
    weather_from = pd.read_csv('../Data/train/output/weather_airport_vec.csv')
    weather_from.columns = [u'出发机场', u'出发机场天气', u'出发机场最低气温', u'出发机场最高气温', 'date', 'weatherVecFrom']
    weather_from['weatherVecFrom'] = weather_from['weatherVecFrom'].apply(lambda x: [float(i) for i in x[1:-1].split(', ')])
    weather_from['weatherVecFrom0'] = weather_from['weatherVecFrom'].apply(lambda x: x[0])
    weather_from['weatherVecFrom1'] = weather_from['weatherVecFrom'].apply(lambda x: x[1])
    weather_from['weatherVecFrom2'] = weather_from['weatherVecFrom'].apply(lambda x: x[2])
    
    # 到达机场天气
    weather_to = pd.read_csv('../Data/train/output/weather_airport_vec.csv')
    weather_to.columns = [u'到达机场', u'到达机场天气', u'到达机场最低气温', u'到达机场最高气温', 'date', 'weatherVecTo']
    weather_to['weatherVecTo'] = weather_to['weatherVecTo'].apply(lambda x: [float(i) for i in x[1:-1].split(', ')])
    weather_to['weatherVecTo0'] = weather_to['weatherVecTo'].apply(lambda x: x[0])
    weather_to['weatherVecTo1'] = weather_to['weatherVecTo'].apply(lambda x: x[1])
    weather_to['weatherVecTo2'] = weather_to['weatherVecTo'].apply(lambda x: x[2])
    print('Load weather data success')

    # 机场特情
    sn = load_special_news()
    print('Load special news data success')

    # 添加天气特征
    data_from = pd.merge(df, weather_from, on=[u'出发机场', 'date'], how='inner')
    data = pd.merge(data_from, weather_to, on=[u'到达机场', 'date'], how='inner')
    print('Extract weather feature success')

    # 添加特情
    data['hasSpecialNews'] = 0
    for index, row in sn.iterrows():
        # print data.ix[0,u'计划起飞时间'] - row[u'结束时间']
        data[((data[u'出发机场'] == row[u'特情机场']) | (data[u'到达机场'] == row[u'特情机场'])) & (
            (data[u'计划起飞时间'] > row[u'开始时间']) & (data[u'计划起飞时间'] < row[u'结束时间']))]['hasSpecialNews'] = 1
        # break
    print('Extract special news feature success')

    # 缺失值 
    data = data.dropna()
    data.to_csv('../Feature/sample_data_with_all_feature.csv',index=False,encoding='gbk')
    # print data[data['hasSpecialNews'] == True]
    # print data[data['timePrepareThisFlightPlan'] != 0].head()
    print('Load data with all feature success')
    
    
    return data

def load_sample_data_with_feature():
    df = pd.read_csv('../Feature/sample_data_with_feature.csv',encoding='gbk')
    return df

def train_model(trainX, trainY):
    # 训练分类模型

    # 逻辑斯特回归
    # m = LogisticRegression(penalty='l2',C=1000.0, random_state=0)

    # 随机森林
    m = RandomForestRegressor(min_samples_split=210, min_samples_leaf=20, max_depth=25, random_state=10)
    # m = RandomForestRegressor()

    # SVM
    # m = svm.SVC(kernel='rbf')
    # m = svm.SVC(kernel='rbf', C = 1, probability=True, random_state=0)

    m.fit(trainX, trainY)
    return m


def evaluate(real_v, predict_v):
    # 得出评价指标
    AUC = metrics.roc_auc_score(real_v, predict_v)
    print("AUC: %f" % AUC)

    threshold = 0.5
    for i in range(0, len(predict_v)):
        if predict_v[i] > threshold:
            predict_v[i] = 1
        else:
            predict_v[i] = 0
    confusion_matrix = metrics.confusion_matrix(real_v, predict_v)

    tp = confusion_matrix[0][0]
    tn = confusion_matrix[1][1]
    fp = confusion_matrix[1][0]
    fn = confusion_matrix[0][1]

    # print tp, fn, fp,tn


    acc = float(tp + tn) / (tp + tn + fp + fn)
    recall = float(tp) / (tp + fn)
    pre = float(tp) / (tp + fp)

    print("ACC: ", acc)
    print("Pre: ", pre)
    print("Recall: ", recall)

    print
    confusion_matrix
    return [AUC, acc, pre, recall]

    '''
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(real_v, predict_v)
    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)



    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)
    plt.show()
    '''


def log(p, record):
    log_file = open(p, 'a')
    log_file.write(record + "\n")
    log_file.close()
    
    
def model_cmd():
    
    res_path = "res_0-22.txt"
    print("==> Modeling start ")
    log_file = open(res_path, 'w')
    log_file.write("--> start\n")
    log_file.close()

    # 获取正负样本数据
    data = load_sample_data_with_feature()

    NUM = len(data)
    count_data_n = len(data[data['isMoreThan3'] == 1])
    print('正样本数量: %d' % (NUM - count_data_n))
    print('负样本数量: %d' % count_data_n)
    log(res_path, "==> get positive samples: " + str(NUM - count_data_n))
    log(res_path, "==> get negative samples: " + str(count_data_n))
    print("总数据： ", NUM)
    
    Feature = [
        # 航班基本信息
        # u'航班编号',
        u'计划起飞时间',
        u'计划到达时间',
        u'飞机编号',
        u'flightTime',

        # 航班历史延误信息
        u'平均延误时间',
        u'延误时间标准差',
        u'最大延误时间',
        u'延误时间中位数',

        # 前序航班
        # u'lastFlight',
        u'timeLastFlightDelay',
        u'timePrepareThisFlightRemain',
        u'timePrepareThisFlightPlan',
    
        # 天气
        u'出发机场最低气温',
        u'出发机场最高气温',
        u'weatherVecFrom0',
        u'weatherVecFrom1',
        u'weatherVecFrom2',
        u'到达机场最低气温',
        u'到达机场最高气温',
        u'weatherVecTo0',
        u'weatherVecTo1',
        u'weatherVecTo2',

        # 特情
        u'hasSpecialNews'
    ]
    Label = ['isMoreThan3']

    skf = KFold(NUM, n_folds=10, shuffle=True)
    count = 0
    AUC_set = []
    acc_set = []
    pre_set = []
    recall_set = []
    f1_set = []
    print("---------  Modeling ------------")
    log(res_path, "---------  Modeling ------------")
    print("--------------------------------")
    log(res_path, "--------------------------------")
    for train, test in skf:
        print("--> fold: ", count)
        log(res_path, "Fold: " + str(count))

        train_x = data.ix[train, Feature]
        train_x = Imputer().fit_transform(train_x)
        train_y = data.ix[train, Label]

        test_x = data.ix[test, Feature]
        test_y = data.ix[test, Label]

        # 模型训练
        m = train_model(train_x, train_y)

        # 训练集AUC
        y_predprob = m.predict(train_x)
        train_auc = metrics.roc_auc_score(train_y, y_predprob)
        print("AUC Score (train): %f" % train_auc)
        log(res_path, "| AUC (Train):\t" + str(train_auc))

        # 模型测试
        y_predprob = m.predict(test_x)
        [AUC, acc, pre, recall] = evaluate(test_y, y_predprob)
        AUC_set.append(AUC)
        acc_set.append(acc)
        pre_set.append(pre)
        recall_set.append(recall)

        f1 = 2 * pre * recall / (pre + recall)
        f1_set.append(f1)
        print("F1: ", f1)
        log(res_path, "| AUC Score (test):\t" + str(AUC))
        log(res_path,
            "| Accuracy: " + str(acc) + " | Precision: " + str(pre) + " | Recall: " + str(recall) + " | F1: " + str(f1))
        count += 1
        print("--------------------------------")
        log(res_path, "--------------------------------")

    mean_auc = sum(AUC_set) / len(AUC_set)
    print("Mean AUC: ", mean_auc)
    log(res_path, "Mean AUC: " + str(mean_auc))
    mean_acc = sum(acc_set) / len(acc_set)
    print("Mean acc: ", mean_acc)
    log(res_path, "Mean acc: " + str(mean_acc))
    mean_pre = sum(pre_set) / len(pre_set)
    print("Mean pre: ", mean_pre)
    log(res_path, "Mean pre: " + str(mean_pre))
    mean_recall = sum(recall_set) / len(recall_set)
    print("Mean recall: ", mean_recall)
    log(res_path, "Mean recall: " + str(mean_recall))
    mean_f1 = sum(f1_set) / len(f1_set)
    print("Mean F1: ", mean_f1)
    log(res_path, "Mean F1: " + str(mean_f1))
    
    print('-----------------------------')
    log(res_path, "--------------------------------")
    
    test = load_test_data()
    p = m.predict(test.ix[:,Feature])
    print(p)
    log(res_path, str(p))
    
    print("over!")


if __name__ == '__main__':
    
    # load_test_data()
    
    model_cmd()
    # load_special_news()
    # load_data()
    # data = load_data()
    # load_weather()
