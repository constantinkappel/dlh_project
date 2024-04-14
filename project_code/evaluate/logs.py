import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import *

## Read log files

def get_trainlog_paths(root: Union[str, Path]):
    root = Path(root)
    return list(root.glob("*/*/train.log"))

def read_log(path: Union[str, Path]):
    with open(path, "r") as f:
        lines = f.readlines()
    return lines

## Parse log files for hyperparameters

def create_pattern(token: str):
    return re.compile(r"'{}': '(\w+)'".format(token))

def parse_log(train_log: List[str], tokens: List[str]):
    hyperparams = {}
    for line in train_log:
        for tok in tokens:
            pattern = create_pattern(tok)
            match = pattern.search(line)
            if match:
                #print(match.group(1))
                hyperparams[tok] = match.group(1)
    return hyperparams


def parse_experiment(root: Union[str, Path], tokens: List[str], patterns: Dict[str, re.Pattern]=None):
    l_trainlogs = get_trainlog_paths(root)
    df = pd.DataFrame()
    for log in l_trainlogs:
        train_log = read_log(log)
        hyperparams = parse_log(train_log, tokens)
        run = log.parent.name
        hyperparams['run'] = log.parent#/run
        if "done training" in train_log[-1]:
            hyperparams['done'] = True
        else:
            hyperparams['done'] = False
        if patterns:
            hyperparams['auprc'] = get_last_auprc(train_log, patterns)
        df = pd.concat([df, pd.DataFrame(hyperparams, index=[0])])
    if patterns:
        df['auprc'] = df['auprc'].astype(float)
        order = ['run', 'done', 'src_data', 'task', 'embed_model', 'model', 'value_mode', 'auprc']
    else:
        order = ['run', 'done', 'src_data', 'task', 'embed_model', 'model', 'value_mode']
    return df[order]

## parse metric values

def get_last_auprc(train_log: List[str], patterns: Dict[str, re.Pattern]):
    for line in train_log[::-1]:
        hit = patterns['auprc'].search(line)
        if hit:
            return hit.group(1)
        
def create_pattern_numerical(token: str):
    return re.compile(r"{}: (\d+(\.\d+)?)".format(token))

def extract_metrics(line: str, df_metrics: pd.DataFrame, patterns: Dict[str, re.Pattern], fold: str):
    epoch = patterns['epoch'].search(line).group(1)
    loss = patterns['loss'].search(line).group(1)
    auroc = patterns['auroc'].search(line).group(1)
    auprc = patterns['auprc'].search(line).group(1)
    df = pd.DataFrame({'fold': fold, 'epoch': int(epoch), 'loss': float(loss), 'auroc': float(auroc), 'auprc': float(auprc)}, index=[0])
    df_metrics = pd.concat([df_metrics, df])
    return df_metrics


def parse_experiment_metrics(root: Union[str, Path], patterns: Dict[str, re.Pattern]):
    l_trainlogs = get_trainlog_paths(root)
    df_metrics = pd.DataFrame(columns=['run', 'fold', 'epoch', 'loss', 'auroc', 'auprc'])
    for log in l_trainlogs:
        train_log = read_log(log)
        run = log.parent
        for line in train_log:
            if "[INFO]" in line:
                if "[train]" in line:
                    df_metrics = extract_metrics(line, df_metrics, patterns, "train")
                    df_metrics['run'] = run
                elif "[valid]" in line:
                    df_metrics = extract_metrics(line, df_metrics, patterns, "valid")
                    df_metrics['run'] = run
                elif "[test]" in line:
                    df_metrics = extract_metrics(line, df_metrics, patterns, "test")
                    df_metrics['run'] = run
    return df_metrics

## Plotting etc. 

def plot_metrics(df_metrics: pd.DataFrame, run: str):
    fig, ax = plt.subplots()
    for fold in ['train', 'valid', 'test']:
        df_metrics.loc[(df_metrics['fold']==fold) & (df_metrics['run']==run)].plot(x='epoch', y='loss', ax=ax, label=fold)
    plt.title(run)
    plt.show()