import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
from preProcess import loadPerformances, loadSpecs

import pdb

def processSpecNorm(data):
    data = data.iloc[125:576, :]
    for i in range(data.shape[1]):
        coe = data.iloc[:, i].mean()
        data.iloc[:, i] = data.iloc[:, i] / coe

    return data

def processSpecDiff(data):
    data = data.diff(2)
    data = data.iloc[125:576, :]

    return data

def processSpec(df_spec, method="normal"):
    if method == "normal":
        df = processSpecNorm(df_spec)
    elif method == "diff":
        df = processSpecDiff(df_spec)
    else:
        df = df_spec
        print("method name {} is wrong".format(method))
    return df

def cal_coe(df_spec, performances, index_name="COD"):
    performance = performances.loc[index_name]
    coefs = []
    for i in range(df_spec.shape[0]):
        coef = np.corrcoef(df_spec.iloc[i], performance)[0, 1]
        coefs.append(coef)
    return np.array(coefs)

def parserExp(df_spec, coefs, exps, k=3):
    '''
    exp: expression on max and min,
    example (a[0] + a[-1]) / a[-2], 0 = max , -1 = min
    '''
    indSorted= np.argsort(coefs)
    inds = np.concatenate([indSorted[:k], indSorted[-k:]])[::-1]
    a = df_spec.iloc[inds].values
    newdata = []
    for exp in exps:
        newdata.append(eval(exp))

    indexs = df_spec.index[inds]
    index_new = []
    pattern = "\[-[0-9]*\]"
    nind = 2*k
    for exp in exps:
        print("old", exp)
        ms = re.finditer(pattern, exp)
        for m in ms:
            print(m)
            print(m.start(0)+1, m.end(0)-1)
            exp = exp.replace(exp[m.start(0):m.end(0)],
                        "[%d]" %(2*k + int(exp[m.start(0)+1:m.end(0)-1])))
        print("new", exp)

        indNeg= exp.find("[=")
        exp = exp.replace("a[", "{").replace("]", "}")
        index_new.append(exp.format(*indexs))

    return pd.DataFrame(newdata, columns=df_spec.columns, index=index_new)

def genNewdata(df_spec, coefs, exps):
    newdata = parserExp(df_spec_prepro, coefs, exps)
    coefs_new = cal_coe(newdata, df_performance, index)
    print("correlation of new data {} is {}".format(exps, coefs_new))
    return newdata, coefs_new

def modelSpecPer(df_spec, df_performance, index_name,
                 model="linear",
                 train_ratio=0.9):
    # row df_spec is data of spec, can be processed
    inds = np.arange(df_spec.shape[1])
    np.random.seed(5)
    np.random.shuffle(inds)
    ntrain = int(len(inds) * train_ratio)
    inds_train = inds[:ntrain]
    inds_test = inds[ntrain:]
    if model == "linear":
        y = df_performance.loc[index_name].values[inds_train]
        for i in range(df_spec.shape[0]):
            x = df_spec.iloc[0].values[inds_train]
            rep = np.polyfit(x, y, 1)
            func = np.poly1d(rep)
            y_fit = func(x)
            plt.figure(figsize=(18, 10))
            plt.scatter(x, y)
            plt.plot(x, y_fit, label=df_spec.index[i])
            plt.legend(loc="upper right")
            plt.savefig("%s_%d.png" %(model, i))


if __name__ == "__main__":
    datapaths_spec = ["./data/spec_data/caiyang1/csv/", "./data/spec_data/caiyang2/csv"]
    datapaths_per = ["./data/实测指标/caiyang1_zb.csv", "./data/实测指标/caiyang2_zb.csv"]
    df_spec = loadSpecs(datapaths_spec)
    df_performance = loadPerformances(datapaths_per)
    assert df_spec.shape[1] == df_performance.shape[1], \
        "spectrum and performance has diff sample num"

    method = "normal" # method for process spec
    exps = ["(a[0]+a[1]) / 2", "(a[0]+a[-1]) / 2"] # expression for new feature
    index = "COD" # performance index name

    df_spec_prepro = processSpec(df_spec, method=method)
    coefs = cal_coe(df_spec_prepro, df_performance, index)
    # plot coefs vs wavelength
    df_spec_new, coefs_new = genNewdata(df_spec_prepro, coefs, exps)

    modelSpecPer(df_spec_new, df_performance, "COD")
