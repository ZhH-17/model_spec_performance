import numpy as np
import pandas as pd
import scipy.io as sio
from functools import partial
import os
import pdb

def loadCsv(fn):
    data = np.loadtxt(fn, skiprows=13)
    with open(fn, 'r') as fr:
        [fr.readline() for i in range(3)]
        exposeTime = int(fr.readline().split(":")[1].strip()[:-1])
        datatype = fr.readline().split(":")[1].strip()[1:-2]
    return data, exposeTime, datatype


def reflectfn(fn, white=None):
    # fn: csv file, white: raw data of white board, normalized by time
    data, exposeTime, datatype = loadCsv(fn)
    if datatype == "REF_TYPE":
        # reflect data
        return data[:, 1] / data[:, 2]
    elif datatype == "RAW_TYPE":
        # raw data
        if white is None:
            # white board data
            return data[:, 1] / exposeTime
        else:
            return data[:, 1] / exposeTime / white
    else:
        print("csv data type error!")
        return None


def loadSpec(datapath, r=1.):
    # df: wavelengths * samples
    prefixFn = "Spectrum%05d.csv"
    wavelengths = np.arange(325, 1076)

    descriptionFn = os.path.join(datapath, "data.txt")
    assert os.path.exists(descriptionFn), "not exist data description file"
    with open(descriptionFn, 'r') as fr:
        lines = fr.readlines()
    suffix_white, suffix_samples, suffix_sky = \
        [list(map(int, line.split(" "))) for line in lines[:3]]
    assert len(suffix_white) == 1
    assert len(suffix_sky) == 1
    print("in {}:\n white board suffix {}\n samples suffix {}\n sky suffix {}"
          .format(datapath, suffix_white, suffix_samples, suffix_sky))

    white = reflectfn(os.path.join(datapath, prefixFn %suffix_white[0]))

    fns_samples = [prefixFn %i for i in suffix_samples]
    ref_samples = [reflectfn(os.path.join(datapath, fn), white) for fn in fns_samples]

    ref_sky = reflectfn(os.path.join(datapath, prefixFn %suffix_sky[0]), white)
    ref_samples = ref_samples - r * ref_sky

    ref_samples = np.array(ref_samples).T
    df = pd.DataFrame(ref_samples,
                      columns=suffix_samples,
                      index=wavelengths)
    return df

def loadSpecs(datapaths, r=1.):
    # df:
    wavelengths = np.arange(325, 1076)
    df = pd.DataFrame(index=wavelengths)
    for ind, datapath in enumerate(datapaths):
        # basename = os.path.basename(datapath)
        basename = "no.%d_" %ind
        dfi = loadSpec(datapath)
        dfi.rename(columns=lambda x: basename+str(x), inplace=True)
        dfi.to_csv(os.path.join(datapath, "ref.csv"))
        df = pd.concat([df, dfi], axis=1)
    return df

def loadPerformance(datapath):
    df = pd.read_csv(datapath, delimiter=',')
    print("load performance data from {}".format(datapath))
    return df

def loadPerformances(datapaths):
    # df: performance * samples
    dfs = [loadPerformance(datapath) for datapath in datapaths]
    return pd.concat(dfs).T

if __name__ == "__main__":
    datapaths_spec = ["./data/spec_data/caiyang1/csv/", "./data/spec_data/caiyang2/csv"]
    datapaths_per = ["./data/实测指标/caiyang1_zb.csv",
                     "./data/实测指标/caiyang2_zb.csv"]
    df_spec = loadSpecs(datapaths_spec)
    df_performance = loadPerformances(datapaths_per)
