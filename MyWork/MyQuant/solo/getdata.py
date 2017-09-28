# -*- coding:utf-8 -*-

import tushare as ts
import talib
import pandas as pd
import numpy as np
from sklearn import preprocessing


def data_recieve(name, start, data_type='normal', ktype='D'):
    if data_type not in {'normal', 'carg', 'grow'}:
        raise ValueError('Invalid data_type "%s"' % data_type)

    data = ts.get_k_data(name, start=start, ktype=ktype)

    data_var = data.drop(data.columns, axis=1)
    date = data.pop('date').values
    code = data.pop('code')

    #print('获得原始数据：\t %s' % code[0])
    print('起始时间：\t %s' % str(date[0]))
    print('结束时间：\t %s' % str(date[-1]))
    print('数据个数：\t %s\n' % len(date))
    print data.head(5)

    if data_type == 'normal':
        for i in data.columns:
            data_var[i] = data[i]

    if data_type == 'carg':
        for i in data.columns:
            data_var[i] = (data[i] - data[i].shift(1)) / data[i].shift(1)

    if data_type == 'grow':
        for i in data.columns:
            data_var[i] = data[i] / data[i].shift(1)

    open = data_var.open.values
    high = data_var.high.values
    close = data_var.close.values
    low = data_var.low.values
    volume = data_var.volume.values



    new_data = data.drop(data.columns, axis=1)
    new_data.index = date
    new_data['open'] = open
    new_data['high'] = high
    new_data['close'] = close
    new_data['low'] = low
    new_data['volume'] = volume
    new_data = new_data.dropna()

    print('\n查看最新数据：\n')
    print new_data.head(5)

    return new_data


def data_indicator(data,time,normal=False):

    ml_datas = data.drop(data.columns, axis=1)

    open = data.open.values
    high = data.high.values
    close = data.close.values
    low = data.low.values
    volume = data.volume.values
    var = [open,high,close,low,volume]
    var_name = ['open','high','close','low','volume']




    # 单输入带时间单输出
    #为了凑数，以下候补
    #[talib.DEMA, talib.WMA, talib.MAXINDEX, talib.MININDEX, talib.TEMA ]
    #["DEMA", "WMA", "MAXINDEX", "MININDEX", "TEMA"]


    single = [talib.EMA, talib.KAMA, talib.MA, talib.MIDPOINT, talib.SMA, talib.T3, talib.TRIMA,
              talib.CMO, talib.MOM, talib.ROC, talib.ROCP, talib.ROCR, talib.ROCR100, talib.RSI, talib.TRIX, talib.MAX,
              talib.MIN, talib.SUM]
    single_name = ["EMA", "KAMA", "MA", "MIDPOINT", "SMA", "T3", "TRIMA", "CMO", "MOM", "ROC", "ROCP",
                   "ROCR", "ROCR100", "RSI", "TRIX", "MAX", "MIN", "SUM"]

    def single_output(f, x1, timeperiod):
        z = f(x1, timeperiod)
        return z

    for i in time:
        for v in range(len(var)):
            for p in range(len(single)):
                locals()[single_name[p] + str('_') + var_name[v] + str('_') + str(i)] = single_output(single[p], var[v],
                                                                                                      timeperiod=i)


    for i in time:
        for v in range(len(var)):
            for p in range(len(single)):
                ml_datas[single_name[p] + str('_') + var_name[v] + str('_') + str(i)] = pd.Series(
                    locals()[single_name[p] + str('_') + var_name[v] + str('_') + str(i)], index=data.index)


    #单输入带时间多输出
    for i in time:
        for v in range(len(var)):
            locals()['BBANDS_upper'+ str('_') + var_name[v] + str('_') + str(i)], \
            locals()['BBANDS_middle' + str('_') + var_name[v] + str('_') + str(i)],\
            locals()['BBANDS_lower'+ str('_') + var_name[v] + str('_') + str(i)] = talib.BBANDS(var[v], timeperiod=i)

            locals()['STOCHRSI_fastk' + str('_') + var_name[v] + str('_') + str(i)], \
            locals()['STOCHRSI_fastd' + str('_') + var_name[v] + str('_') + str(i)] = talib.STOCHRSI(var[v], timeperiod=i)

            locals()['MINMAX_min' + str('_') + var_name[v] + str('_') + str(i)], \
            locals()['MINMAX_max' + str('_') + var_name[v] + str('_') + str(i)] = talib.MINMAX(var[v], timeperiod=i)

            locals()['MINMAX_minidx' + str('_') + var_name[v] + str('_') + str(i)], \
            locals()['MINMAX_maxidx' + str('_') + var_name[v] + str('_') + str(i)] = talib.MINMAXINDEX(var[v], timeperiod=i)



    for i in time:
        for v in range(len(var)):
            ml_datas['BBANDS_upper'+ str('_') + var_name[v] + str('_') + str(i)] = pd.Series(
                locals()['BBANDS_upper'+ str('_') + var_name[v] + str('_') + str(i)], index=data.index)
            ml_datas['BBANDS_middle' + str('_') + var_name[v] + str('_') + str(i)] = pd.Series(
                locals()['BBANDS_middle' + str('_') + var_name[v] + str('_') + str(i)], index=data.index)
            ml_datas['BBANDS_lower' + str('_') + var_name[v] + str('_') + str(i)] = pd.Series(
                locals()['BBANDS_lower' + str('_') + var_name[v] + str('_') + str(i)], index=data.index)
            ml_datas['STOCHRSI_fastk' + str('_') + var_name[v] + str('_') + str(i)] = pd.Series(
                locals()['STOCHRSI_fastk' + str('_') + var_name[v] + str('_') + str(i)], index=data.index)
            ml_datas['STOCHRSI_fastd' + str('_') + var_name[v] + str('_') + str(i)] = pd.Series(
                locals()['STOCHRSI_fastd' + str('_') + var_name[v] + str('_') + str(i)], index=data.index)
            ml_datas['MINMAX_min' + str('_') + var_name[v] + str('_') + str(i)] = pd.Series(
                locals()['MINMAX_min' + str('_') + var_name[v] + str('_') + str(i)], index=data.index)
            ml_datas['MINMAX_max' + str('_') + var_name[v] + str('_') + str(i)] = pd.Series(
                locals()['MINMAX_max' + str('_') + var_name[v] + str('_') + str(i)], index=data.index)
            ml_datas['MINMAX_minidx' + str('_') + var_name[v] + str('_') + str(i)] = pd.Series(
               locals()['MINMAX_minidx' + str('_') + var_name[v] + str('_') + str(i)], index=data.index)
            ml_datas['MINMAX_maxidx' + str('_') + var_name[v] + str('_') + str(i)] = pd.Series(
                locals()['MINMAX_maxidx' + str('_') + var_name[v] + str('_') + str(i)], index=data.index)

    # 多输入带时间单输出
    for i in time:
        locals()['ATR' + str('_') + str(i)] = talib.ATR(high, low, close, timeperiod=i)
        locals()['NATR' + str('_') + str(i)] = talib.NATR(high, low, close, timeperiod=i)
        locals()['ADX' + str('_') + str(i)] = talib.ADX(high, low, close, timeperiod=i)
        locals()['ADXR' + str('_') + str(i)] = talib.ADXR(high, low, close, timeperiod=i)
        locals()['AROONOSC' + str('_') + str(i)] = talib.AROONOSC(high, low, timeperiod=i)
        locals()['CCI' + str('_') + str(i)] = talib.CCI(high, low, close, timeperiod=i)
        locals()['DX' + str('_') + str(i)] = talib.DX(high, low, close, timeperiod=i)
        locals()['MFI' + str('_') + str(i)] = talib.MFI(high, low, close, volume, timeperiod=i)
        locals()['MINUS_DI' + str('_') + str(i)] = talib.MINUS_DI(high, low, close, timeperiod=i)
        locals()['MINUS_DM' + str('_') + str(i)] = talib.MINUS_DM(high, low, timeperiod=i)
        locals()['PLUS_DI' + str('_') + str(i)] = talib.PLUS_DI(high, low, close, timeperiod=i)
        locals()['PLUS_DM' + str('_') + str(i)] = talib.PLUS_DM(high, low, timeperiod=i)
        locals()['WILLR' + str('_') + str(i)] = talib.WILLR(high, low, close, timeperiod=i)
        locals()['MIDPRICE' + str('_') + str(i)] = talib.MIDPRICE(high, low, timeperiod=i)
        locals()['AROON_aroondown' + str('_') + str(i)], locals()['AROON_aroonup' + str('_') + str(i)] = talib.AROON(high, low, timeperiod=i)

    for i in time:
        ml_datas['ATR'] = pd.Series(locals()['ATR' + str('_') + str(i)], index=data.index)
        ml_datas['NATR'] = pd.Series(locals()['NATR' + str('_') + str(i)], index = data.index)
        ml_datas['ADX'] = pd.Series(locals()['ADX' + str('_') + str(i)], index = data.index)
        ml_datas['ADXR'] = pd.Series(locals()['ADXR' + str('_') + str(i)], index = data.index)
        ml_datas['AROONOSC'] = pd.Series(locals()['AROONOSC' + str('_') + str(i)], index = data.index)
        ml_datas['CCI'] = pd.Series(locals()['CCI' + str('_') + str(i)], index = data.index)
        ml_datas['DX'] = pd.Series(locals()['DX' + str('_') + str(i)], index = data.index)
        ml_datas['MFI'] = pd.Series(locals()['MFI' + str('_') + str(i)], index = data.index)
        ml_datas['MINUS_DI'] = pd.Series(locals()['MINUS_DI' + str('_') + str(i)], index = data.index)
        ml_datas['MINUS_DM'] = pd.Series(locals()['MINUS_DM' + str('_') + str(i)], index = data.index)
        ml_datas['PLUS_DI'] = pd.Series(locals()['PLUS_DI' + str('_') + str(i)], index = data.index)
        ml_datas['PLUS_DM'] = pd.Series(locals()['PLUS_DM' + str('_') + str(i)], index = data.index)
        ml_datas['WILLR'] = pd.Series(locals()['WILLR' + str('_') + str(i)], index = data.index)
        ml_datas['MIDPRICE'] = pd.Series(locals()['MIDPRICE' + str('_') + str(i)], index = data.index)
        ml_datas['AROON_aroondown'] = pd.Series(locals()['AROON_aroondown' + str('_') + str(i)], index = data.index)
        ml_datas['AROON_aroonup'] = pd.Series(locals()['AROON_aroonup' + str('_') + str(i)], index = data.index)




    #单输入不带时间
    # single2 = [talib.ACOS, talib.ASIN, talib.ATAN, talib.CEIL, talib.COS, talib.COSH, talib.EXP, talib.FLOOR, talib.LN,
    #            talib.LOG10, talib.SIN, talib.SINH, talib.SQRT, talib.TAN, talib.TANH, talib.HT_DCPERIOD,
    #            talib.HT_DCPHASE, talib.HT_TRENDMODE, talib.HT_TRENDLINE, talib.APO]
    # single2_name = ["ACOS", "ASIN", "ATAN", "CEIL", "COS", "COSH", "EXP", "FLOOR", "LN", "LOG10", "SIN", "SINH", "SQRT",
    #                 "TAN", "TANH", "HT_DCPERIOD", "HT_DCPHASE", "HT_TRENDMODE", "HT_TRENDLINE", "APO"]
    #
    # def single2_output(f, x1):
    #     z = f(x1)
    #     return z
    #
    # for v in range(len(var)):
    #     for p in range(len(single2)):
    #         locals()[single2_name[p] + str('_') + var_name[v]] = single2_output(single2[p], var[v])
    #
    # for v in range(len(var)):
    #     for p in range(len(single2)):
    #         ml_datas[single2_name[p] + str('_') + var_name[v]] = pd.Series(locals()[single2_name[p] + str('_') + var_name[v]])
    #
    #






    # 模式识别类指标

    pattern = [talib.CDL2CROWS, talib.CDL3BLACKCROWS, talib.CDL3INSIDE, talib.CDL3LINESTRIKE, talib.CDL3OUTSIDE,
               talib.CDL3STARSINSOUTH, talib.CDL3WHITESOLDIERS, talib.CDLABANDONEDBABY, talib.CDLADVANCEBLOCK,
               talib.CDLBELTHOLD, talib.CDLBREAKAWAY, talib.CDLCLOSINGMARUBOZU, talib.CDLCONCEALBABYSWALL,
               talib.CDLCOUNTERATTACK, talib.CDLDARKCLOUDCOVER, talib.CDLDOJI, talib.CDLDOJISTAR,
               talib.CDLDRAGONFLYDOJI,
               talib.CDLENGULFING, talib.CDLEVENINGDOJISTAR, talib.CDLEVENINGSTAR, talib.CDLGAPSIDESIDEWHITE,
               talib.CDLGRAVESTONEDOJI, talib.CDLHAMMER, talib.CDLHANGINGMAN, talib.CDLHARAMI, talib.CDLHARAMICROSS,
               talib.CDLHIGHWAVE, talib.CDLHIKKAKE, talib.CDLHIKKAKEMOD, talib.CDLHOMINGPIGEON,
               talib.CDLIDENTICAL3CROWS,
               talib.CDLINNECK, talib.CDLINVERTEDHAMMER, talib.CDLKICKING, talib.CDLKICKINGBYLENGTH,
               talib.CDLLADDERBOTTOM,
               talib.CDLLONGLEGGEDDOJI, talib.CDLLONGLINE, talib.CDLMARUBOZU, talib.CDLMATCHINGLOW, talib.CDLMATHOLD,
               talib.CDLMORNINGDOJISTAR, talib.CDLMORNINGSTAR, talib.CDLONNECK, talib.CDLPIERCING, talib.CDLRICKSHAWMAN,
               talib.CDLRISEFALL3METHODS, talib.CDLSEPARATINGLINES, talib.CDLSHOOTINGSTAR, talib.CDLSHORTLINE,
               talib.CDLSPINNINGTOP, talib.CDLSTALLEDPATTERN, talib.CDLXSIDEGAP3METHODS, talib.CDLSTICKSANDWICH,
               talib.CDLTAKURI, talib.CDLTASUKIGAP, talib.CDLTHRUSTING, talib.CDLTRISTAR, talib.CDLUNIQUE3RIVER, talib.CDLUPSIDEGAP2CROWS]
    pattern_name = ["CDL2CROWS", "CDL3BLACKCROWS", "CDL3INSIDE", "CDL3LINESTRIKE", "CDL3OUTSIDE", "CDL3STARSINSOUTH",
                "CDL3WHITESOLDIERS", "CDLABANDONEDBABY", "CDLADVANCEBLOCK", "CDLBELTHOLD", "CDLBREAKAWAY",
                "CDLCLOSINGMARUBOZU", "CDLCONCEALBABYSWALL", "CDLCOUNTERATTACK", "CDLDARKCLOUDCOVER", "CDLDOJI",
                "CDLDOJISTAR", "CDLDRAGONFLYDOJI", "CDLENGULFING", "CDLEVENINGDOJISTAR", "CDLEVENINGSTAR",
                "CDLGAPSIDESIDEWHITE", "CDLGRAVESTONEDOJI", "CDLHAMMER", "CDLHANGINGMAN", "CDLHARAMI", "CDLHARAMICROSS",
                "CDLHIGHWAVE", "CDLHIKKAKE", "CDLHIKKAKEMOD", "CDLHOMINGPIGEON", "CDLIDENTICAL3CROWS", "CDLINNECK",
                "CDLINVERTEDHAMMER", "CDLKICKING", "CDLKICKINGBYLENGTH", "CDLLADDERBOTTOM", "CDLLONGLEGGEDDOJI",
                "CDLLONGLINE", "CDLMARUBOZU", "CDLMATCHINGLOW", "CDLMATHOLD", "CDLMORNINGDOJISTAR", "CDLMORNINGSTAR",
                "CDLONNECK", "CDLPIERCING", "CDLRICKSHAWMAN", "CDLRISEFALL3METHODS", "CDLSEPARATINGLINES",
                "CDLSHOOTINGSTAR", "CDLSHORTLINE", "CDLSPINNINGTOP", "CDLSTALLEDPATTERN","CDLXSIDEGAP3METHODS","CDLSTICKSANDWICH","CDLTAKURI", "CDLTASUKIGAP", "CDLTHRUSTING", "CDLTRISTAR", "CDLUNIQUE3RIVER", "CDLUPSIDEGAP2CROWS"]


    def Pattern_Recognition(f, x1, x2, x3, x4):
            z = f(x1, x2, x3, x4)
            return z


    for p in range(len(pattern)):
        locals()[pattern_name[p]] = Pattern_Recognition(pattern[p], open, high, low, close)

    for p in range(len(pattern)):
        ml_datas[pattern_name[p]] = pd.Series(locals()[pattern_name[p]], index=data.index)


    #杂乱指标
    #为了凑数，ULTOSC多用了一遍

    ADD = talib.ADD(high, low)
    MULT = talib.MULT(high, low)
    SUB = talib.SUB(high, low)
    TRANGE = talib.TRANGE(high, low, close)
    AD = talib.AD(high, low, close, volume)
    ADOSC = talib.ADOSC(high, low, close, volume)
    OBV = talib.OBV(close, volume)
    BOP = talib.BOP(open, high, low, close)

    ml_datas['ADD'] = pd.Series(ADD, index=data.index)
    ml_datas['MULT'] = pd.Series(MULT, index=data.index)
    ml_datas['SUB'] = pd.Series(SUB, index=data.index)
    ml_datas['TRANGE'] = pd.Series(TRANGE, index=data.index)
    ml_datas['AD'] = pd.Series(AD, index=data.index)
    ml_datas['ADOSC'] = pd.Series(ADOSC, index=data.index)
    ml_datas['OBV'] = pd.Series(OBV, index=data.index)
    ml_datas['BOP'] = pd.Series(BOP, index=data.index)


    HT_PHASOR_inphase, HT_PHASOR_quadrature = talib.HT_PHASOR(close)
    HT_SINE_sine, HT_SINE_leadsine = talib.HT_SINE(close)
    MACD_macd, MACD_macdsignal, MACD_macdhist = talib.MACD(close)
    MACDEXT_macd, MACDEXT_macdsignal, MACDEXT_macdhist = talib.MACDEXT(close)
    MACDFIX_macd, MACDFIX_macdsignal, MACDFIX_macdhist = talib.MACDFIX(close)
    PPO = talib.PPO(close)
    MAMA_mama, MAMA_fama = talib.MAMA(close)
    STOCH_slowk, STOCH_slowd = talib.STOCH(high, low, close)
    STOCHF_fastk, STOCHF_fastd = talib.STOCHF(high, low, close)
    SAR = talib.SAR(high, low)
    SAREXT = talib.SAREXT(high, low)
    ULTOSC = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)


    ml_datas['HT_PHASOR_inphase'] = pd.Series(HT_PHASOR_inphase, index=data.index)
    ml_datas['HT_PHASOR_quadrature'] = pd.Series(HT_PHASOR_quadrature, index=data.index)
    ml_datas['HT_SINE_sine'] = pd.Series(HT_SINE_sine, index=data.index)
    ml_datas['HT_SINE_leadsine'] = pd.Series(HT_SINE_leadsine, index=data.index)
    ml_datas['MACD_macd'] = pd.Series(MACD_macd, index=data.index)
    ml_datas['MACD_macdsignal'] = pd.Series(MACD_macdsignal, index=data.index)
    ml_datas['MACD_macdhist'] = pd.Series(MACD_macdhist, index=data.index)
    ml_datas['MACDEXT_macd'] = pd.Series(MACDEXT_macd, index=data.index)
    ml_datas['MACDEXT_macdsignal'] = pd.Series(MACDEXT_macdsignal, index=data.index)
    ml_datas['MACDEXT_macdhist'] = pd.Series(MACDEXT_macdhist, index=data.index)
    ml_datas['MACDFIX_macd'] = pd.Series(MACDFIX_macd, index=data.index)
    ml_datas['MACDFIX_macdsignal'] = pd.Series(MACDFIX_macdsignal, index=data.index)
    ml_datas['MACDFIX_macdhist'] = pd.Series(MACDFIX_macdhist, index=data.index)
    ml_datas['PPO'] = pd.Series(PPO, index=data.index)
    ml_datas['MAMA_mama'] = pd.Series(MAMA_mama, index=data.index)
    ml_datas['MAMA_fama'] = pd.Series(MAMA_fama, index=data.index)
    ml_datas['STOCH_slowk'] = pd.Series(STOCH_slowk, index=data.index)
    ml_datas['STOCH_slowd'] = pd.Series(STOCH_slowd, index=data.index)
    ml_datas['STOCHF_fastk'] = pd.Series(STOCHF_fastk, index=data.index)
    ml_datas['STOCHF_fastd'] = pd.Series(STOCHF_fastd, index=data.index)
    ml_datas['SAR'] = pd.Series(SAR, index=data.index)
    ml_datas['SAREXT'] = pd.Series(SAREXT, index=data.index)
    ml_datas['ULTOSC'] = pd.Series(ULTOSC, index=data.index)
    ml_datas['ULTOSC_VAR'] = pd.Series(ULTOSC, index=data.index)






    # 将原始数据集的数据移动一天，使每天收盘价数据的特征训练的时候用前一天的信息
    ml_datas = ml_datas.shift(1)
    ml_datas['target'] = close*100



    #var_datas = ml_datas.drop(ml_datas.columns, axis=1)

    #var_datas['target'] = var_datas.sum(axis=1) * 100

    #ml_datas['target'] = var_datas['target']




    #ml_datas = ml_datas.dropna(how='all', axis=1) #删掉都是NA的列
    ml_datas = ml_datas.dropna(how='any', axis=0)

    if normal:
        X_ori = ml_datas.drop(['target'], axis=1)
        scaler = preprocessing.StandardScaler().fit(X_ori)
        X = scaler.transform(X_ori)
        X_ori = pd.DataFrame(X,index=X_ori.index,columns=X_ori.columns)

        format = lambda x: '%.1f' % x
        X_ori['target'] = ml_datas['target'].map(format).map(float) #保留n位小数，然后转回float

        #X_ori['target'] = pd.Series(ml_datas['target'],dtype=str)

        ml_datas = X_ori.copy()

    return ml_datas



def data_trans(data,data_y,days=1,depth=28):
    XX = np.zeros([data.shape[0]-days+1, days, depth, data.shape[1] / depth])
    YY = np.zeros([data.shape[0]-days+1])
    data = data.reshape(data.shape[0], depth, data.shape[1] / depth)
    for i in range(data.shape[0]-days+1):
        XX[i:,:,:] = data[i :i+days,:,:]

    YY = data_y[days-1:]
    return XX,YY