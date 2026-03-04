import matplotlib
import os
import pickle
import random
import re
import shutil
import time
import torch
import yaml

from itertools import product
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return {"result": result, "time": end_time - start_time}
    return wrapper


def load_npy(path):
    return np.load(path)


def load_pkl(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


def load_yaml_as_df(path):
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
    df = pd.json_normalize(data)
    return df


def exist_metric(exp_dir):
    try:
        metric_path = os.path.join(exp_dir, 'performance.yaml')
        if os.path.exists(metric_path):
            return True
        else:
            return False
    except Exception as e:
        return False


def exist_pred(exp_dir):
    try:
        
        res_dir = os.path.join(exp_dir, 'results')
        setting = os.listdir(res_dir)[0]
        setting_dir = os.path.join(res_dir, setting)
        metric_path = os.path.join(setting_dir, 'pred.npy')
        if os.path.exists(metric_path):
            return True, setting_dir
        else:
            return False, None
    except Exception as e:
        return False, None


def exist_stf_metric(exp_dir):
    try:
        res_dir = os.path.join(exp_dir, 'results')
        metric_dir = os.path.join(res_dir, "m4_results")
        metric_path = os.path.join(metric_dir, 'metrics.pkl')
        if os.path.exists(metric_path):
            settings = os.listdir(res_dir)
            setting = [s for s in settings if 'Hourly' in s][0]
            setting_dir = os.path.join(res_dir, setting)
            return True, metric_dir, setting_dir
        else:
            return False, None, None
    except Exception as e:
        return False, None, None


def inverse_stf_metrics(metrics, names):
    new_metrics = {}
    for key, value in metrics.items():
        if key not in names:
            continue
        for k, v in value.items():
            if k in new_metrics:
                new_metrics[k][key] = v
            else:
                new_metrics[k] = {key: v}
    return new_metrics


def keep_split(exp, special_words=[]):
    pattern = '|'.join(map(re.escape, special_words)) + '|_'
    parts = re.findall(f'({pattern})|([^_]+)', exp)
    result = [part[0] or part[1] for part in parts if any(part) and (part[0] or part[1]) != '_']
    output = []
    for part in result:
        try: part = eval(part)
        except: pass
        output.append(part)
    return output


def is_full_group(x):
    if str(x['data_id'].iloc[0]).startswith('PEMS'):
        return set(x['pred_len']) == {12, 24, 36, 48}
    else:
        return set(x['pred_len']) == {96, 192, 336, 720}


def load_metric_from_log(log_path):
    with open(log_path, 'r') as f:
        lines = f.readlines()
    for line in lines[::-1]:
        if line.strip() == '':
            continue

        if 'mse' in line and 'mae' in line:
            parts = line.strip().split(', ')
            metrics = {}
            for part in parts:
                key_value = part.split(':')
                if len(key_value) == 2:
                    key = key_value[0].strip()
                    try:
                        value = float(key_value[1].strip())
                    except ValueError:
                        value = key_value[1].strip()
                    metrics[key] = value
            return metrics
    return None


def extract_log_loss(log_file_path):
    """
    从指定路径的日志文件中读取并提取 Epoch, Train loss 和 Val loss。

    Args:
        log_file_path (str): 日志文件的路径 (例如: 'logs/std.log')

    Returns:
        dict: 包含 'epoch', 'train_loss', 'val_loss' 列表的字典。
              如果文件不存在，返回 None。
    """

    if not os.path.exists(log_file_path):
        print(f"错误: 找不到文件 {log_file_path}")
        return None

    metrics = {
        "epoch": [],
        "train_loss": [],
        "val_loss": []
    }

    pattern = re.compile(r"Epoch\s+(\d+)/\d+,\s+Train loss:\s+([\d.]+),\s+Val loss:\s+([\d.]+)")
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    epoch = int(match.group(1))
                    t_loss = float(match.group(2))
                    v_loss = float(match.group(3))

                    metrics["epoch"].append(epoch)
                    metrics["train_loss"].append(t_loss)
                    metrics["val_loss"].append(v_loss)
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return None
    return metrics


if __name__ == '__main__':

    log_path = '/mnt/tidalfs-bdsz01/dataset/llm_dataset/plc_data/TrustworthRLHF/causal-rm/results/baselines_binary/naive_0.0005_512_256,64_1e-6_600_30_saferlhf_0.1_42_train_true_0.1_0.2_1.0/stdout.log'
    losses = extract_log_loss(log_path)
    print(np.mean(losses['val_loss']))