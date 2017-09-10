# coding=utf-8

from __future__ import print_function
import pandas as pd
import datetime, time
from gensim.models import word2vec


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

    fsock = open("../Data/test A/output/weather_word.txt", "w")
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
    df.to_csv('../Data/test A/output/weather_airport_vec.csv', index=False, encoding='gbk')


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
            # chunks.append(chunk)
            chunks.append(extractBasicFeature(chunk))
            print(count * chunkSize, 'Extract basic feature success')
            # break
        except StopIteration:
            loop = False

    df = pd.concat(chunks, ignore_index=True)
    # print(len(df[u'飞机编号'].drop_duplicates())) 3410
    df.to_csv('../Feature/data_with_basic_feature.csv', index=False, encoding='gbk')

    df = extractFlightFeature(df)
    df.to_csv('../Feature/data_with_flight_feature.csv', index=False, encoding='gbk')

    df = extractLastFlightFeature(df)
    df.to_csv('../Feature/data_with_last_flight_feature.csv', index=False, encoding='gbk')

    df = balanceSample(df)
    print('Balance sample success')

    # 出发机场天气
    weather_from = pd.read_csv('../Data/train/output/weather_airport_vec.csv')
    weather_from.columns = [u'出发机场', u'出发机场天气', u'出发机场最低气温', u'出发机场最高气温', 'date', 'weatherVecFrom']
    weather_from['weatherVecFrom'] = weather_from['weatherVecFrom'].apply(
        lambda x: [float(i) for i in x[1:-1].split(', ')])
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


def load_test_data(test_data_path, label):
    # 获取数据
    '''
    df = pd.read_csv('../Data/test A/flight_information.csv',encoding="gbk")

    df = extractBasicFeature(df)
    df = extractFlightFeature(df)        
    df = extractLastFlightFeature(df)
    df = extractNeedSubmitData(df)

    df.to_csv('../Data/test A/output/data_feature.csv',index=False,encoding='gbk')
    '''
    #    df = pd.read_csv('../Data/test A/output/data_feature.csv', encoding="gbk")
    df = pd.read_csv(test_data_path, encoding='gbk')
    load_test_weather()

    # 出发机场天气
    weather_from = pd.read_csv('../Data/test A/output/weather_airport_vec.csv')
    weather_from.columns = [u'出发机场', u'出发机场天气', u'出发机场最低气温', u'出发机场最高气温', 'date', 'weatherVecFrom']
    weather_from = weather_from.drop_duplicates([u'出发机场', 'date'])
    weather_from['weatherVecFrom'] = weather_from['weatherVecFrom'].apply(
        lambda x: [float(i) for i in x[1:-1].split(', ')])
    weather_from['weatherVecFrom0'] = weather_from['weatherVecFrom'].apply(lambda x: x[0])
    weather_from['weatherVecFrom1'] = weather_from['weatherVecFrom'].apply(lambda x: x[1])
    weather_from['weatherVecFrom2'] = weather_from['weatherVecFrom'].apply(lambda x: x[2])

    # 到达机场天气
    weather_to = pd.read_csv('../Data/test A/output/weather_airport_vec.csv')
    weather_to.columns = [u'到达机场', u'到达机场天气', u'到达机场最低气温', u'到达机场最高气温', 'date', 'weatherVecTo']
    weather_to = weather_to.drop_duplicates([u'到达机场', 'date'])
    weather_to['weatherVecTo'] = weather_to['weatherVecTo'].apply(lambda x: [float(i) for i in x[1:-1].split(', ')])
    weather_to['weatherVecTo0'] = weather_to['weatherVecTo'].apply(lambda x: x[0])
    weather_to['weatherVecTo1'] = weather_to['weatherVecTo'].apply(lambda x: x[1])
    weather_to['weatherVecTo2'] = weather_to['weatherVecTo'].apply(lambda x: x[2])
    print('Load weather data success')

    # 机场特情
    sn = load_test_special_news()
    print('Load special news data success')
    if label == 'no_lastflight_no_weather' or label == 'has_lastflight_no_weather':
        # 添加特情
        df['hasSpecialNews'] = 0
        for index, row in sn.iterrows():
            # print data.ix[0,u'计划起飞时间'] - row[u'结束时间']
            df.loc[((df.loc[:, u'出发机场'] == row[u'特情机场']) | (df.loc[:, u'到达机场'] == row[u'特情机场'])) & (
                (df.loc[:, u'计划起飞时间'] > row[u'开始时间']) & (df.loc[:, u'计划起飞时间'] < row[u'结束时间'])), ['hasSpecialNews']] = 1
            # break
        print('Extract special news feature success')

        print('Load ' + label + ' data with feature success')
        return df

    elif label == 'no_lastflight_has_weather' or label == 'has_lastflight_has_weather':
        # 添加天气特征
        data_from = pd.merge(df, weather_from, on=[u'出发机场', 'date'], how='inner')
        data = pd.merge(data_from, weather_to, on=[u'到达机场', 'date'], how='inner')
        print('Extract weather feature success')
        # 添加特情
        data['hasSpecialNews'] = 0
        for index, row in sn.iterrows():
            # print data.ix[0,u'计划起飞时间'] - row[u'结束时间']
            data.loc[((data.loc[:, u'出发机场'] == row[u'特情机场']) | (data.loc[:, u'到达机场'] == row[u'特情机场'])) & (
                (data.loc[:, u'计划起飞时间'] > row[u'开始时间']) & (data.loc[:, u'计划起飞时间'] < row[u'结束时间'])), [
                         'hasSpecialNews']] = 1
            # break
        print('Extract special news feature success')

        print('Load ' + label + ' data with feature success')
        return data

        # # 添加天气特征
        # data_from = pd.merge(df, weather_from, on=[u'出发机场', 'date'], how='inner')
        # print(data_from.shape)
        # data = pd.merge(data_from, weather_to, on=[u'到达机场', 'date'], how='inner')
        # print(data.shape)
        # print('Extract weather feature success')
        #
        # # 添加特情
        # data['hasSpecialNews'] = 0
        # for index, row in sn.iterrows():
        #     # print data.ix[0,u'计划起飞时间'] - row[u'结束时间']
        #     data[((data[u'出发机场'] == row[u'特情机场']) | (data[u'到达机场'] == row[u'特情机场'])) & (
        #         (data[u'计划起飞时间'] > row[u'开始时间']) & (data[u'计划起飞时间'] < row[u'结束时间']))]['hasSpecialNews'] = 1
        #     # break
        # print('Extract special news feature success')
        #
        # # 缺失值
        # data = data.dropna()
        #
        # # print data[data['hasSpecialNews'] == True]
        # # print data[data['timePrepareThisFlightPlan'] != 0].head()
        # print('Load test data with feature success')
        # return data


def extract_avg_delay():
    # 获取数据
    reader = pd.read_csv('../Data/train/flight_information.csv', encoding="gbk")
    # 缺失值
    df = reader.dropna()

    dict = {}
    for flight in df.groupby([u'出发机场']):
        # 机场编号
        key = flight[0]
        values = flight[1]

        # 该机场平均延误时间
        avg_delay_time = (values[u'实际起飞时间'] - values[u'计划起飞时间']).mean()
        dict[key] = avg_delay_time

    data = pd.DataFrame(list(dict.items()), columns=[u'出发机场', 'avg_delay_time'])
    data.to_csv('../Data/train/output/flight_avg_delay_time.csv', index=False, encoding='gbk')
    print(data)
    print("==> extract avg delay time success.")


def classify_test_data():
    weather = pd.read_csv('../Data/test A/output/weather_airport_vec.csv', encoding='gbk')
    print(weather.head())

    # 构建天气集合用于查询
    weather_set = set()
    for index, value in weather.iterrows():
        key_from = str(value[u'城市']) + "||" + str(value[u'日期'])
        weather_set.add(key_from)

    test_data = pd.read_csv('../Data/test A/output/data_feature.csv', encoding='gbk')

    # 首先对飞机编号进行填充,统一填充成0
    test_data = test_data.fillna({u'飞机编号': 0})

    # 获得列名
    cols = test_data.columns.values.tolist()

    # 缺少前序航班没天气的
    no_lastflight_no_weather = []
    # 缺少前序航班有天气的
    no_lastflight_has_weather = []
    # 有前序航班和无天气的
    has_lastflight_no_weather = []
    # 啥都有
    has_lastflight_has_weather = []
    for index, value in test_data.iterrows():
        # 没有前序航班的
        if pd.isnull(value[u'lastFlight']):

            # 没有前序航班没有天气向量
            key_from = str(value[u'出发机场']) + "||" + str(value[u'date'])
            key_to = str(value[u'到达机场']) + "||" + str(value[u'date'])
            if (key_from not in weather_set) or (key_to not in weather_set):
                no_lastflight_no_weather.append(value)
            else:
                no_lastflight_has_weather.append(value)
        # 有前序航班
        else:

            key_from = str(value[u'出发机场']) + "||" + str(value[u'date'])
            key_to = str(value[u'到达机场']) + "||" + str(value[u'date'])
            # 有前序航班没天气的
            if (key_from not in weather_set) or (key_to not in weather_set):
                has_lastflight_no_weather.append(value)
            # 有前序航班有天气的
            else:
                has_lastflight_has_weather.append(value)

    no_lastflight_no_weather_df = pd.DataFrame(no_lastflight_no_weather, columns=cols)
    no_lastflight_no_weather_df.to_csv('../Data/test A/output/no_lastflight_no_weather.csv', index=False,
                                       encoding='gbk')
    no_lastflight_has_weather_df = pd.DataFrame(no_lastflight_has_weather, columns=cols)
    no_lastflight_has_weather_df.to_csv('../Data/test A/output/no_lastflight_has_weather.csv', index=False,
                                        encoding='gbk')
    has_lastflight_no_weather_df = pd.DataFrame(has_lastflight_no_weather, columns=cols)
    has_lastflight_no_weather_df.to_csv('../Data/test A/output/has_lastflight_no_weather.csv', index=False,
                                        encoding='gbk')
    has_lastflight_has_weather_df = pd.DataFrame(has_lastflight_has_weather, columns=cols)
    has_lastflight_has_weather_df.to_csv('../Data/test A/output/has_lastflight_has_weather.csv', index=False,
                                         encoding='gbk')

    print("===> classify test data finished.")


def extractBasicFeature(df):
    # 加载平均延迟
    avg_delay = pd.read_csv('../Data/train/output/flight_avg_delay_time.csv', encoding='gbk')

    # 所有机场的平均延时
    avg_mean = avg_delay['avg_delay_time'].mean()

    # 构造字典
    dict = {}
    dict['avg_mean'] = avg_mean
    for index, values in avg_delay.iterrows():
        dict[values[u'出发机场']] = values['avg_delay_time']

    print("==>load flight_avg_delay_time.csv success.")

    # 填补

    for index, values in df.iterrows():
        curr_key = values[u'出发机场']
        if (pd.isnull(values[u'实际起飞时间'])):
            if (dict.has_key(curr_key)):
                df.ix[index, [u'实际起飞时间']] = values[u'计划起飞时间'] + int(dict[curr_key])
                df.ix[index, [u'实际到达时间']] = values[u'计划到达时间'] + int(dict[curr_key])
            else:
                df.ix[index, [u'实际起飞时间']] = values[u'计划起飞时间'] + int(dict['avg_mean'])
                df.ix[index, [u'实际到达时间']] = values[u'计划到达时间'] + int(dict['avg_mean'])

        if (pd.isnull(values[u'航班是否取消'])):
            df.ix[index, [u'航班是否取消']] = u'否'

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

        # print(flight[0], ' Extract flight feature success')

        describe = flight[1]['delay'].describe()
        if len(flight[1]) > 1:
            flight_feature.append(
                [flight[0][0], flight[0][1], describe['mean'], describe['std'], describe['max'], describe['50%']])
        else:
            flight_feature.append([flight[0][0], flight[0][1], describe['mean'], 0, describe['max'], describe['50%']])
    df_flight_feature = pd.DataFrame(flight_feature,
                                     columns=[u'航班编号', u'出发机场', u'平均延误时间', u'延误时间标准差', u'最大延误时间', u'延误时间中位数'])
    df = pd.merge(df, df_flight_feature, on=[u'航班编号', u'出发机场'], how='left')
    return df


def extractLastFlightFeature(df):
    # 添加前序航班特征
    dfg_all = []

    # 对于有缺失值的列填充
    df_na = df[df[u'飞机编号'].isnull()]
    df_na['lastFlight'] = ''
    df_na['timeLastFlightDelay'] = 0
    df_na['timePrepareThisFlightRemain'] = 0
    df_na['timePrepareThisFlightPlan'] = 0
    dfg_all.append(df_na)

    # 去掉缺失值
    df = df.dropna()

    for flight in df.groupby([u'飞机编号', u'date']):

        # print(flight[0], ' Extract last flight feature success')

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


def extractNeedSubmitData(df):
    print("==> extract data which we should submit.")
    df = df[df[u'需验证标识（1为需提交结果、0不需要提交）'].isin(['1'])]
    return df


def build_submission_result(predict_X, predict_Y):
    print("==>prepare for submission.")
    submission = pd.DataFrame(columns=[])
    submission.loc[:, 'Flightno'] = predict_X.loc[:, u'航班编号']
    submission.loc[:, 'FlightDepcode'] = predict_X.loc[:, u'出发机场']
    submission.loc[:, 'FlightArrcode'] = predict_X.loc[:, u'到达机场']
    submission.loc[:, 'PlannedDeptime'] = predict_X.loc[:, u'计划起飞时间'].astype('int')
    submission.loc[:, 'PlannedArrtime'] = predict_X.loc[:, u'计划到达时间'].astype('int')
    submission.loc[:, 'prob'] = predict_Y.loc[:, 'prob']
    print("==>build submission result finished.")
    return submission


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
            print(count * chunkSize, 'Load data with feature success')
            # break
        except StopIteration:
            loop = False

    df = pd.concat(chunks, ignore_index=True)
    df = balanceSample(df)
    print('Balance sample success')

    # 出发机场天气
    weather_from = pd.read_csv('../Data/train/output/weather_airport_vec.csv')
    weather_from.columns = [u'出发机场', u'出发机场天气', u'出发机场最低气温', u'出发机场最高气温', 'date', 'weatherVecFrom']
    weather_from['weatherVecFrom'] = weather_from['weatherVecFrom'].apply(
        lambda x: [float(i) for i in x[1:-1].split(', ')])
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
    data.to_csv('../Feature/sample_data_with_all_feature.csv', index=False, encoding='gbk')
    # print data[data['hasSpecialNews'] == True]
    # print data[data['timePrepareThisFlightPlan'] != 0].head()
    print('Load data with all feature success')

    return data


def load_sample_data_with_feature():
    df = pd.read_csv('../Feature/sample_data_with_feature.csv', encoding='gbk')
    return df


if __name__ == '__main__':
    classify_test_data()
    no_lastflight_no_weather = load_test_data('../Data/test A/output/no_lastflight_no_weather.csv',
                                              'no_lastflight_no_weather')
    print(no_lastflight_no_weather.shape)
    no_lastflight_has_weather = load_test_data('../Data/test A/output/no_lastflight_has_weather.csv',
                                               'no_lastflight_has_weather')
    print(no_lastflight_has_weather.shape)
    has_lastflight_no_weather = load_test_data('../Data/test A/output/has_lastflight_no_weather.csv',
                                               'has_lastflight_no_weather')
    print(has_lastflight_no_weather.shape)
    has_lastflight_has_weather = load_test_data('../Data/test A/output/has_lastflight_has_weather.csv',
                                                'has_lastflight_has_weather')
    print(has_lastflight_has_weather.shape)
