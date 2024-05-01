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
    return re.compile(r"'{}': '?(\w+)'?".format(token))

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

def get_last_metric(train_log: List[str], metric: str, patterns: Dict[str, re.Pattern]):
    for line in train_log[::-1]:
        hit = patterns[metric].search(line) # e.g. metric = 'auprc'
        if hit:
            return hit.group(1)

def parse_experiment(root: Union[str, Path], tokens: List[str], patterns: Dict[str, re.Pattern]=None):
    l_trainlogs = get_trainlog_paths(root)
    df = pd.DataFrame()
    for log in tqdm(l_trainlogs):
        train_log = read_log(log)
        hyperparams = parse_log(train_log, tokens)
        run = log.parent.name
        hyperparams['run'] = log.parent#/run
        if "done training" in train_log[-1]:
            hyperparams['done'] = True
        else:
            hyperparams['done'] = False
        if patterns:
            for metric in patterns.keys():
                hyperparams[metric] = get_last_metric(train_log, metric, patterns)
        df = pd.concat([df, pd.DataFrame(hyperparams, index=[0])])
    if patterns:
        for metric in patterns.keys():
            if metric in df.columns:
                df[metric] = df[metric].astype(float)
        order = tokens + list(patterns.keys())
    else:
        order = tokens
    order = [col for col in order if col in df.columns]# Return dataframe columns which exist 
    return df[order]

## parse metric values



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

def tag_experiment(df: pd.DataFrame) -> pd.DataFrame:
    df['tag'] = len(df) * ['unknown']
    # CodeEmb RD & W2V
    df.loc[(df['embed_model'] == 'codeemb') & (df['load_pretrained_weights'] == False), "tag"] = "CodeEmb_RD"
    df.loc[(df['embed_model'] == 'codeemb') & (df['load_pretrained_weights'] == True), "tag"] = "CodeEmb_W2V"
    # All transfer experiments
    df.loc[(df['embed_model'] == 'codeemb') & (df['transfer'] == True), "tag"] = "CodeEmb_Transfer"
    df.loc[(df['embed_model'] == 'descemb_bert') & (df['transfer'] == True), "tag"] = "DescEmb-BERT_Transfer"
    df.loc[(df['embed_model'] == 'descemb_rnn') & (df['transfer'] == True), "tag"] = "DescEmb-RNN_Transfer"
    # All pooled experiments
    df.loc[(df['embed_model'] == 'codeemb') & (df['src_data'] == 'pooled'), "tag"] = "CodeEmb-Pooled"
    df.loc[(df['embed_model'] == 'descemb_bert') & (df['src_data'] == 'pooled'), "tag"] = "DescEmb-BERT_Pooled"
    df.loc[(df['embed_model'] == 'descemb_rnn') & (df['src_data'] == 'pooled'), "tag"] = "DescEmb-RNN_Pooled"
    # All MLM pretraining experiments
    df.loc[(df['task'] == 'w2v') & (df['model'] == 'codeemb'), "tag"] = "CodeEmb_Pretrain-W2V"
    df.loc[(df['task'] == 'mlm') & (df['model'] == 'descemb_bert') & (df['init_bert_params'] == True), "tag"] = "DescEmb-BERT_Pretrain-MLM"
    df.loc[(df['task'] == 'mlm') & (df['model'] == 'descemb_rnn') & (df['init_bert_params'] == True), "tag"] = "DescEmb-RNN_Pretrain-MLM"
    # CLS-FT
    df.loc[(df['model'] == 'ehr_model') & (df['embed_model'] == 'descemb_bert') & (df['init_bert_params_with_freeze'] == True), "tag"] = "DescEmb-BERT_CLS-FT"
    # FT
    df.loc[(df['model'] == 'ehr_model') & (df['embed_model'] == 'descemb_bert') & (df['init_bert_params'] == True), "tag"] = "DescEmb-BERT_FT"
    # FT-MLM
    df.loc[(df['model'] == 'ehr_model') & (df['embed_model'] == 'descemb_bert') & (df['load_pretrained_weights'] == True), "tag"] = "DescEmb-BERT_FT-MLM"
    # RNN Scr & Scr-MLM
    df.loc[(df['model'] == 'ehr_model') & (df['embed_model'] == 'descemb_rnn') & (df['load_pretrained_weights'] == False), "tag"] = "DescEmb-RNN_Scr"
    df.loc[(df['model'] == 'ehr_model') & (df['embed_model'] == 'descemb_rnn') & (df['load_pretrained_weights'] == True), "tag"] = "DescEmb-RNN_Scr"
    return df

## Plotting etc. 

def plot_metrics(df_metrics: pd.DataFrame, run: Union[str|List[str]], metrics: List[str] = ['loss', 'auroc', 'auprc'], folds: str = ['train', 'valid', 'test']):
    ncols = len(metrics)
    if isinstance(run, str):
        run = [run]
    nrows = len(run)
    height = 5*nrows
    width = 3*5
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, height))
    axes = axes.ravel() if nrows > 1 else axes
    for j, r in enumerate(run):
        for i, metric in enumerate(metrics):
            for fold in folds:
                ax = axes[j*len(metrics) + i] if ncols > 1 else axes
                df_metrics.loc[(df_metrics['fold']==fold) & (df_metrics['run']==r)].plot(x='epoch', y=metric, ax=ax, label=fold)
                ax.set_title(metric)
    fig.suptitle(run)
    plt.tight_layout()
    plt.show()