# coding=utf-8

from __future__ import print_function
import pandas as pd
import os
from sklearn.externals import joblib

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

# flight 加载数据方法
import flight

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


def model_cmd(op=0):
    
    global m
    res_path = "res_0-22.txt"
    print("==> Modeling start ")
    log_file = open(res_path, 'w')
    log_file.write("--> start\n")
    log_file.close()

    # 获取正负样本数据
    data = flight.load_sample_data_with_feature()

    NUM = len(data)
    count_data_n = len(data[data['isMoreThan3'] == 1])
    print('正样本数量: %d' % (NUM - count_data_n))
    print('负样本数量: %d' % count_data_n)
    log(res_path, "==> get positive samples: " + str(NUM - count_data_n))
    log(res_path, "==> get negative samples: " + str(count_data_n))
    print("总数据： ", NUM)

    # 基础特征
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

        # 特情
        u'hasSpecialNews'
    ]
    
    test_data = flight.load_test_data('../Data/test A/output/no_lastflight_no_weather.csv',
                                              'no_lastflight_no_weather')
    
    # 基础特征 + 前序航班 
    if op == 1:
        Feature.extend([                    
            # u'lastFlight',
            u'timeLastFlightDelay',
            u'timePrepareThisFlightRemain',
            u'timePrepareThisFlightPlan'
        ])
        
        test_data = flight.load_test_data('../Data/test A/output/has_lastflight_no_weather.csv',
                                               'has_lastflight_no_weather')
    
    # 基础特征 + 天气
    if op == 2:
        Feature.extend([                    
            u'出发机场最低气温',
            u'出发机场最高气温',
            u'weatherVecFrom0',
            u'weatherVecFrom1',
            u'weatherVecFrom2',
            u'到达机场最低气温',
            u'到达机场最高气温',
            u'weatherVecTo0',
            u'weatherVecTo1',
            u'weatherVecTo2'
        ])
        
        test_data = flight.load_test_data('../Data/test A/output/no_lastflight_has_weather.csv',
                                               'no_lastflight_has_weather')
        
    # 基础特征 + 前序航班 + 天气
    if op == 3:
        Feature.extend([ 
            # u'lastFlight',
            u'timeLastFlightDelay',
            u'timePrepareThisFlightRemain',
            u'timePrepareThisFlightPlan',                   
            u'出发机场最低气温',
            u'出发机场最高气温',
            u'weatherVecFrom0',
            u'weatherVecFrom1',
            u'weatherVecFrom2',
            u'到达机场最低气温',
            u'到达机场最高气温',
            u'weatherVecTo0',
            u'weatherVecTo1',
            u'weatherVecTo2'
        ])
        
        test_data = flight.load_test_data('../Data/test A/output/has_lastflight_has_weather.csv',
                                                'has_lastflight_has_weather')
        
    Label = ['isMoreThan3']

    skf = KFold(NUM, n_folds=10, shuffle=True)
    count = 0
    AUC_set = []
    acc_set = []
    pre_set = []
    recall_set = []
    f1_set = []
    total_probability = pd.DataFrame(columns=[])
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
        m = []

        if (os.path.isfile('../Model/' + str(op) + 'm' + str(count) + '.model')):
            # 模型加载
            m = joblib.load('../Model/' + str(op) + 'm' + str(count) + '.model')
            print('==> ' + str(op) + 'm' + str(count) + '.model had been trained.')
            log(res_path, '==> ' + str(op) + 'm' + str(count) + '.model had been trained.')
        else:
            # 模型训练
            m = train_model(train_x, train_y)
            print('==> ' + str(op) + 'm' + str(count) + '.model has been training.')
            log(res_path, '==> ' + str(op) + 'm' + str(count) + '.model has been training.')
            joblib.dump(m, '../Model/' + str(op) + 'm' + str(count) + '.model')

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

        #test_data = flight.load_test_data()
        p = m.predict(test_data.ix[:, Feature]).transpose()
        predictDf = pd.DataFrame(p, columns=['prob'])
        if (count == 1):
            total_probability = predictDf
        else:
            total_probability['prob'] += predictDf['prob']
            # print(predictDf.head())
            # log(res_path, str(p))
        print(total_probability.head())

    total_probability['prob'] /= 10
    total_probability = flight.build_submission_result(test_data, total_probability)
    total_probability.to_csv("../Data/test A/output/predict" + str(op) + ".csv", index=False)
    
    print("------------Finished-----------------")
    log(res_path, "------------Finished-----------------")

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
    print("over!")


if __name__ == '__main__':
    
    # model_cmd()
    
    # model_cmd(1)
    
    # model_cmd(2)
    
    # model_cmd(3)
    flight.concat_predict_data()
