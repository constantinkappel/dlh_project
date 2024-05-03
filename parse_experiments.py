import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    from pathlib import Path
    from typing import *
    import project_code.evaluate.logs as logs
    
    
## Global variables
PATH = Path("outputs")
TOKENS = ['run', 'done', 'src_data', 'task', 'value_mode', 'embed_model', 'model', 'load_pretrained_weights', 'init_bert_params', 'init_bert_params_with_freeze', 'transfer']
metrics = ['epoch','loss','auroc', 'auprc']
# create a pattern dictionay with the metrics as keys
PATTERNS = {metric: logs.create_pattern_numerical(metric) for metric in metrics}


if __name__ == "__main__":
    print("Parsing experiments")
    df_experiments = logs.parse_experiment(PATH, TOKENS, PATTERNS)
    df_experiments = logs.tag_experiment(df_experiments)
    print("Saving experiments")
    df_experiments.to_excel(PATH/"experiments.xlsx", index=False)
    print("Parsing metrics")
    df_metrics = logs.parse_experiment_metrics(PATH, PATTERNS)
    print("Saving metrics")
    df_metrics.to_excel(PATH/"metrics.xlsx", index=False)
    print(f"Found {len(df_experiments)} experiments and {len(df_metrics)} metrics.")
    print(df_experiments['tag'].value_counts())