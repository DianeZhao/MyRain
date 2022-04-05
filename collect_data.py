"""
Collect data for benchmark tasks.
"""
import json
import numpy as np
from tqdm import tqdm
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from pathlib import Path
# from src.datasets.scaler import Scaler
from utils import local_time_shift, collate_fn, get_local_shift, is_vbl_const, get_var_name, get_vbl_name
from memmap_dataloader import Dataset


def get_data(hparams: dict, tvt: str = 'train_valid'):
    """Main function to get data according to hparams"""
    datapath, partition_conf, sample_conf = write_data_config(hparams)

    # Collect datasets
    loaderDict = {p: Dataset(datapath=datapath,
                    partition_conf=partition_conf,
                    partition_type="range",
                    partition_selected=p,
                    sample_conf=sample_conf) for p in tvt.split('_')}##train_valid_test

    # Define collate function
    normalizer = read_normalization_stats(hparams['norm_path'])
    if hparams['inc_time']:
        time_shift = get_local_shift(hparams['grid'], loaderDict['train'].dataset)
    collate = lambda x: collate_fn(x, hparams, normalizer, time_shift)

    return hparams, loaderDict, normalizer, collate


def get_checkpoint_path(model_dir):####????????????????????????
    """Return path of latest checkpoint found in the model directory."""
    chkpt =  str(list(Path(model_dir).glob('checkpoints/*'))[-1])
    return chkpt


def read_normalization_stats(path):###??????
    """Read json file storing normalization statistics."""
    with open(path) as f:
        tmp = json.load(f)
    n_dict = {}
    for vbl in tmp:
        n_dict[get_var_name(vbl)] = tmp[vbl]
    return n_dict


def write_partition_conf(sources: str, imerg: bool):######input data configurations:1.dataset source 2.increment time
    """
    Write a time partition configuration dictionary.
    guide for datetime:
    https://www.geeksforgeeks.org/python-datetime-module/
    """
   
    
    if sources in ['simsat', 'simsat_era', 'era16_3']:#sim_sat,simsat_era and era inout data
        #train_timerange = (datetime(2016,4,1,0).timestamp(), datetime(2017, 12, 31,23).timestamp())
        train_timerange = (datetime(2017,12,1,0).timestamp(), datetime(2017, 12, 31,23).timestamp())
        print("train time, from {start} to {end}".format(start=datetime.fromtimestamp(train_timerange[0]),end=datetime.fromtimestamp(train_timerange[1])))
        sample_stride = 3
    
    elif sources == 'era':###useless maybe
        if imerg:
            train_timerange = (datetime(2000,6,1,0).timestamp(), datetime(2017, 12,31,23).timestamp())
        else:
            train_timerange = (datetime(1979,1,1,7).timestamp(), datetime(2017, 12,31,23).timestamp())
        sample_stride = 1

    #val_timerange =  (datetime(2018,1,6,0).timestamp(), datetime(2018, 12,31,23).timestamp())
    #test_timerange =  (datetime(2019,1,6,0).timestamp(), datetime(2019, 12,31,23).timestamp())
    val_timerange =  (datetime(2018,12,6,0).timestamp(), datetime(2018, 12,31,23).timestamp())
    test_timerange =  (datetime(2019,12,10,0).timestamp(), datetime(2019, 12,31,23).timestamp())
    print("validation time, from {start} to {end}".format(start=datetime.fromtimestamp(val_timerange[0]),end=datetime.fromtimestamp(val_timerange[1])))
    print("test time, from {start} to {end}".format(start=datetime.fromtimestamp(test_timerange[0]),end=datetime.fromtimestamp(test_timerange[1])))

    increments = int(sample_stride * 60 * 60)

    partition_conf = {
                        "train":
                            {"timerange": train_timerange,
                            "increment_s": increments},
                        "valid":
                            {"timerange": val_timerange,
                            "increment_s": increments},
                        "test":
                            {"timerange": test_timerange,
                            "increment_s": increments}
                    }
    return partition_conf

  

"""
category
    {'input': inputs,
                'input_temporal': input_temporal,
                'input_temporal_clbt': input_temporal_clbt,
                'input_static': constants,
                'output': output}
    inputs = input_temporal + (['hour', 'day', 'month'] if inc_time else []) + constants
    sample_conf = write_sample_conf(categories, history, lead_times, grid=hparams['grid'])
 """
def write_sample_conf(########
        categories: dict,
        history: list,#NP.ARRANGE()
        lead_times: list,#NP.ARRANGE()
        interporlation: str = "nearest_past",
        grid: float = 5.625):
    """
    Write a sample configuration dictionary.
    """
    sample_conf = {}
##input temporal+time+constants   a list of str
    if 'clbt-0' in categories['input']:#simsat/simsat+era
        samples = {}
        for var in categories['input']:
            if is_vbl_const(var):#if constant variables,then true
                samples[var] = {"vbl": get_vbl_name(var, grid)}
            elif var not in ['hour', 'day', 'month', 'clbt-1', 'clbt-2', 'clbt-0']:#era temporal variables,era5part
                samples[var]  =  {"vbl": get_vbl_name(var, grid), "t": history, "interpolate": interporlation}
            elif var == 'clbt-0':
                samples['clbt']  = {"vbl": get_vbl_name('clbt', grid), "t": history, "interpolate": interporlation}
    else:#only era
        samples = {var: {"vbl": get_vbl_name(var, grid)} if is_vbl_const(var) else \
            {"vbl": get_vbl_name(var, grid), "t": history, "interpolate": interporlation} \
            for var in categories['input'] if var not in ['hour', 'day', 'month']}

    for lt in lead_times:
        sample_conf["lead_time_{}".format(int(lt/3600))] = {
            "label": samples,
            "target": {var: {"vbl": get_vbl_name(var, grid), "t": np.array([lt]), "interpolate": interporlation} \
                for var in categories['output']}
            }
    # output = ['precipitationcal'] if imerg else ['tp']
    return sample_conf

###sample-conf[leadtime1 ,leadtime2,...]
##leadtime-dict{"label"=,"target"=,}

def define_categories(sources: str, inc_time: bool, imerg: bool):
    """
    Write a dictionary which holds lists specifying the model input / output variables.
    """
    simsat_vars_list = ['clbt-0', 'clbt-1', 'clbt-2'] if 'simsat' in sources else []#clbt cloud brightness temprature
    era_vars_list = ['sp', 't2m', 'z-300', 'z-500', 'z-850', 't-300', 't-500', 't-850', \
        'q-300', 'q-500', 'q-850', 'clwc-300', 'clwc-500', 'ciwc-500', 'clwc-850', 'ciwc-850'] if 'era' in sources else []
    simsat_vars_list = ['clbt-0', 'clbt-1', 'clbt-2'] if 'simsat' in sources else []#useless?? defined double times
    simsat_vars_list_clbt = ['clbt'] if 'simsat' in sources else []
    input_temporal = simsat_vars_list + era_vars_list
    input_temporal_clbt = simsat_vars_list_clbt + era_vars_list
    
    constants = ['lsm','orography', 'lat2d', 'lon2d', 'slt']
    inputs = input_temporal + (['hour', 'day', 'month'] if inc_time else []) + constants
    output = ['precipitationcal'] if imerg else ['tp']#####tp???????

    categories = {
                'input': inputs,
                'input_temporal': input_temporal,
                'input_temporal_clbt': input_temporal_clbt,
                'input_static': constants,
                'output': output}

    return categories


def write_data_config(hparams: dict):
    """
    Define configurations for collecting data.
    """
    hparams['latlon'] = (32, 64) if hparams['grid'] == 5.625 else (128, 256)####height and width why not 64,64?????

    # define paths
    datapath = hparams['data_paths']#1

    # define data configurations
    categories = define_categories(hparams['sources'], inc_time=hparams['inc_time'], imerg=hparams['imerg'])
   
    history = np.flip(np.arange(0, hparams['sample_time_window'] + hparams['sample_freq'], hparams['sample_freq']) * -1 * 3600)
    ##sample_time_window,reverse
    lead_times = np.arange(hparams['forecast_freq'], hparams['forecast_time_window'] + hparams['forecast_freq'], hparams['forecast_freq']) * 3600
    ######how to use the leadtime
    partition_conf = write_partition_conf(hparams['sources'], hparams['imerg'])#2
    sample_conf = write_sample_conf(categories, history, lead_times, grid=hparams['grid'])#3

    # define new parameters in hparams
    hparams['categories'] = categories
    hparams['seq_len'] = len(history)
    hparams['forecast_n_steps'] = len(lead_times)
    hparams['out_channels'] = len(categories['output'])
    hparams['num_channels'] = len(categories['input']) + hparams['forecast_n_steps']
    hparams['lead_times'] = lead_times // 3600
    return datapath, partition_conf, sample_conf
