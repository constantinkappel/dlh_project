import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import *
from tqdm import tqdm

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

def extract_metrics(line: str, patterns: Dict[str, re.Pattern]):
    epoch = patterns['epoch'].search(line).group(1)
    loss = patterns['loss'].search(line).group(1)
    auroc = patterns['auroc'].search(line).group(1)
    auprc = patterns['auprc'].search(line).group(1)
    return int(epoch), float(loss), float(auroc), float(auprc)


def parse_experiment_metrics(root: Union[str, Path], patterns: Dict[str, re.Pattern]):
    l_trainlogs = get_trainlog_paths(root)
    print(f"Found {len(l_trainlogs)} train logs.")
    #df_metrics = pd.DataFrame(columns=['run', 'fold', 'epoch', 'loss', 'auroc', 'auprc'])
    dict_metrics = {'run': [], 'fold': [], 'epoch': [], 'loss': [], 'auroc': [], 'auprc': []}
    for log in tqdm(l_trainlogs):
        train_log = read_log(log)
        run = log.parent
        for line in train_log:
            if "[INFO]" in line:
                if "[train]" in line:
                    epoch, loss, auroc, auprc = extract_metrics(line, patterns)
                    dict_metrics['run'].append(run)
                    dict_metrics['fold'].append("train")
                    dict_metrics['epoch'].append(epoch)
                    dict_metrics['loss'].append(loss)
                    dict_metrics['auroc'].append(auroc)
                    dict_metrics['auprc'].append(auprc)
                elif "[valid]" in line:
                    epoch, loss, auroc, auprc = extract_metrics(line, patterns)
                    dict_metrics['run'].append(run)
                    dict_metrics['fold'].append("valid")
                    dict_metrics['epoch'].append(epoch)
                    dict_metrics['loss'].append(loss)
                    dict_metrics['auroc'].append(auroc)
                    dict_metrics['auprc'].append(auprc)
                elif "[test]" in line:
                    epoch, loss, auroc, auprc = extract_metrics(line, patterns)
                    dict_metrics['run'].append(run)
                    dict_metrics['fold'].append("test")
                    dict_metrics['epoch'].append(epoch)
                    dict_metrics['loss'].append(loss)
                    dict_metrics['auroc'].append(auroc)
                    dict_metrics['auprc'].append(auprc)
    df_metrics = pd.DataFrame(dict_metrics)
    return df_metrics

## Plotting etc. 

def plot_metrics(df_metrics: pd.DataFrame, run: str, metrics: List[str] = ['loss', 'auroc', 'auprc'], folds: str = ['train', 'valid', 'test']):
    run = Path(run)
    ncols = len(metrics)
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(15, 5))
    for i, metric in enumerate(metrics):
        for fold in folds:
            ax = axes[i] if ncols > 1 else axes
            df_metrics.loc[(df_metrics['fold']==fold) & (df_metrics['run']==run)].plot(x='epoch', y=metric, ax=ax, label=fold)
            ax.set_title(metric)
    fig.suptitle(run)
    plt.show()