import wandb
import pandas as pd
import os

api = wandb.Api()

metrics_keys = [
    "loss/val",
    "graph_l1000_mcf7/loss/val",
    "graph_l1000_mcf7/avpr/val",
    "graph_l1000_mcf7/auroc/val",
    "graph_l1000_vcap/loss/val",
    "graph_l1000_vcap/avpr/val",
    "graph_l1000_vcap/auroc/val",
    "graph_pcba_1328/loss/val",
    "graph_pcba_1328/auroc/val",
    "graph_pcba_1328/avpr/val",
    "graph_pcqm4m_g25/loss/val",
    "graph_pcqm4m_g25/mae/val",
    "graph_pcqm4m_g25/pearsonr/val",
    "graph_pcqm4m_g25/r2/val",
    "node_pcqm4m_n4/loss/val",
    "node_pcqm4m_n4/mae/val",
    "node_pcqm4m_n4/pearsonr/val",
    "node_pcqm4m_n4/r2/val"
]

class Experiment:
    def __init__(self, wandb_path, model_name, ckpt_select_metric):
        self.wandb_path = wandb_path
        self.model_name = model_name
        self.ckpt_select_metric = ckpt_select_metric
        self.metrics = {key: None for key in metrics_keys}


Experiments = [
    # BugFixOg_2e-1-g25-loss_g25do
    Experiment(model_name='BugFixOg_2e-1-g25-loss_g25do_35M_best',      ckpt_select_metric='loss/val',  wandb_path='research/MolecularFoundationModel_Pretraining/304j0fb5'),
    Experiment(model_name='BugFixOg_2e-1-g25-loss_g25do_35M_last',      ckpt_select_metric='last',      wandb_path='research/MolecularFoundationModel_Pretraining/304j0fb5'),
    Experiment(model_name='BugFixOg_2e-1-g25-loss_g25do_130M_best',     ckpt_select_metric='loss/val',  wandb_path='research/MolecularFoundationModel_Pretraining/g30ce5ux'),
    Experiment(model_name='BugFixOg_2e-1-g25-loss_g25do_130M_last',     ckpt_select_metric='last',      wandb_path='research/MolecularFoundationModel_Pretraining/g30ce5ux'),

    # BugFixOg_layernorm
    Experiment(model_name='BugFixOg_layernorm_10M_best',        ckpt_select_metric='loss/val',  wandb_path='research/MolecularFoundationModel_Pretraining/31diwyje'),
    Experiment(model_name='BugFixOg_layernorm_10M_last',        ckpt_select_metric='last',      wandb_path='research/MolecularFoundationModel_Pretraining/31diwyje'),
    Experiment(model_name='BugFixOg_layernorm_35M_best',        ckpt_select_metric='loss/val',  wandb_path='research/MolecularFoundationModel_Pretraining/40z13f25'),
    Experiment(model_name='BugFixOg_layernorm_35M_last',        ckpt_select_metric='last',      wandb_path='research/MolecularFoundationModel_Pretraining/40z13f25'),
    Experiment(model_name='BugFixOg_layernorm_130M_best',       ckpt_select_metric='loss/val',  wandb_path='research/MolecularFoundationModel_Pretraining/1c83jps3'),
    Experiment(model_name='BugFixOg_layernorm_130M_last',       ckpt_select_metric='last',      wandb_path='research/MolecularFoundationModel_Pretraining/1c83jps3'),
    Experiment(model_name='BugFixOg_layernorm_300M_best',       ckpt_select_metric='loss/val',  wandb_path='research/MolecularFoundationModel_Pretraining/4i0t2jl8'),

    # BugFixOg_5e-1-g25-loss
    Experiment(model_name='BugFixOg_5e-1-g25-loss_10M_best',    ckpt_select_metric='loss/val',  wandb_path='research/MolecularFoundationModel_Pretraining/tffsomax'),
    Experiment(model_name='BugFixOg_5e-1-g25-loss_10M_last',    ckpt_select_metric='last',      wandb_path='research/MolecularFoundationModel_Pretraining/tffsomax'),
    Experiment(model_name='BugFixOg_5e-1-g25-loss_35M_best',    ckpt_select_metric='loss/val',  wandb_path='research/MolecularFoundationModel_Pretraining/due9put8'),
    Experiment(model_name='BugFixOg_5e-1-g25-loss_130M_best',   ckpt_select_metric='loss/val',  wandb_path='research/MolecularFoundationModel_Pretraining/obafj9m9'),
    Experiment(model_name='BugFixOg_5e-1-g25-loss_300M_best',   ckpt_select_metric='loss/val',  wandb_path='research/MolecularFoundationModel_Pretraining/v57hbdix'),

    # BugFixOg 1e-4
    Experiment(model_name='BugFixOg_10M_1e-4_best',             ckpt_select_metric='loss/val',  wandb_path='research/MolecularFoundationModel_Pretraining/xnlulvle'),
    Experiment(model_name='BugFixOg_35M_1e-4_best',             ckpt_select_metric='loss/val',  wandb_path='research/MolecularFoundationModel_Pretraining/wi5sfswn'),
    Experiment(model_name='BugFixOg_130M_1e-4_best',            ckpt_select_metric='loss/val',  wandb_path='research/MolecularFoundationModel_Pretraining/bddckgo1'),
    Experiment(model_name='BugFixOg_300M_1e-4_best',            ckpt_select_metric='loss/val',  wandb_path='research/MolecularFoundationModel_Pretraining/jc7klabw'),
    
    # BugFixOg 3e-4 g25-loss
    Experiment(model_name='BugFixOg_10M_g25-loss',              ckpt_select_metric='loss/val',  wandb_path='research/MolecularFoundationModel_Pretraining/sjuvmmo0'),
    Experiment(model_name='BugFixOg_10M_g25-loss_last',         ckpt_select_metric='last',      wandb_path='research/MolecularFoundationModel_Pretraining/sjuvmmo0'),
    Experiment(model_name='BugFixOg_35M_g25-loss_best',         ckpt_select_metric='loss/val',  wandb_path='research/MolecularFoundationModel_Pretraining/fd3kt53a'),
    Experiment(model_name='BugFixOg_35M_g25-loss_last',         ckpt_select_metric='last',      wandb_path='research/MolecularFoundationModel_Pretraining/fd3kt53a'),
    Experiment(model_name='BugFixOg_130M_g25-loss_best',        ckpt_select_metric='loss/val',  wandb_path='research/MolecularFoundationModel_Pretraining/yjvf7b6b'),
    Experiment(model_name='BugFixOg_130M_g25-loss_last',        ckpt_select_metric='last',      wandb_path='research/MolecularFoundationModel_Pretraining/yjvf7b6b'),
    Experiment(model_name='BugFixOg_300M_g25-loss_best',        ckpt_select_metric='loss/val',  wandb_path='research/MolecularFoundationModel_Pretraining/ss5wbce4'),

    # BugFixOg 2e-4
    Experiment(model_name='BugFixOg_10M',                       ckpt_select_metric='loss/val',  wandb_path='research/MolecularFoundationModel_Pretraining/3kkjbtm3'),
    Experiment(model_name='BugFixOg_10M_last',                  ckpt_select_metric='last',      wandb_path='research/MolecularFoundationModel_Pretraining/3kkjbtm3'),
    Experiment(model_name='BugFixOg_35M',                       ckpt_select_metric='loss/val',  wandb_path='research/MolecularFoundationModel_Pretraining/25f0n0ae'),
    Experiment(model_name='BugFixOg_35M_last',                  ckpt_select_metric='last',      wandb_path='research/MolecularFoundationModel_Pretraining/25f0n0ae'),
    Experiment(model_name='BugFixOg_130M_best',                 ckpt_select_metric='loss/val',  wandb_path='research/MolecularFoundationModel_Pretraining/qgnnjjah'),
    Experiment(model_name='BugFixOg_130M_last',                 ckpt_select_metric='last',      wandb_path='research/MolecularFoundationModel_Pretraining/qgnnjjah'),
    Experiment(model_name='BugFixOg_300M_best',                 ckpt_select_metric='loss/val',  wandb_path='research/MolecularFoundationModel_Pretraining/80fizot6')
]

# Existing data loading
results_csv_path = "results/pretraining_numbers_for_correlation_analysis.csv"
if os.path.exists(results_csv_path):
    df_existing = pd.read_csv(results_csv_path, index_col=0)
else:
    df_existing = pd.DataFrame()

# Initialize a dictionary for the table
table_data = {key: [] for key in metrics_keys}

for exp in Experiments:
    # Skip experiments already processed
    if exp.model_name in df_existing.columns:
        continue
    
    run = api.run(exp.wandb_path)

    best_epoch_val = float('inf')
    best_epoch_idx = None

    # Handle "last" checkpoint selection
    if exp.ckpt_select_metric == 'last':
        history = list(run.scan_history())
        for i in range(len(history) - 1, -1, -1):  # Iterate backwards
            row = history[i]
            if all(key in row for key in exp.metrics.keys()):
                best_epoch_idx = i
                break
    else:
        # Find the best epoch based on the specified metric
        for i, row in enumerate(run.scan_history()):
            value = row.get(exp.ckpt_select_metric, float('inf'))
            if value < best_epoch_val:
                best_epoch_val = value
                best_epoch_idx = i

    # Fill in the metrics for the best epoch
    if best_epoch_idx is not None:
        row = history[best_epoch_idx] if exp.ckpt_select_metric == 'last' else list(run.scan_history())[best_epoch_idx]
        for key in exp.metrics.keys():
            exp.metrics[key] = row.get(key)

    # Add the metrics for this experiment to the table data
    for key in metrics_keys:
        table_data[key].append(exp.metrics.get(key))


if table_data[metrics_keys[0]]:
    df_new = pd.DataFrame(table_data, index=[exp.model_name for exp in Experiments if exp.model_name not in df_existing.columns]).T
    df = pd.concat([df_existing, df_new], axis=1)
else:
    df = df_existing



df.to_csv(results_csv_path)

x_index = df.index

model_names = [exp.model_name for exp in Experiments]
df = df[model_names]

from analyze_results import load_results, BENCHMARKS, MODELS

def save_to_csv(results):
    data = []

    for dataset in BENCHMARKS.keys():
        mean_row = {'Dataset': dataset}

        for (model_name, result_dataset), values in results.items():
            if dataset == result_dataset:
                mean_row[model_name] = values['mean']

        data.append(mean_row)

    df = pd.DataFrame(data)

    df.set_index(['Dataset'], inplace=True)

    ordered_columns = [model for model in MODELS if model in df.columns]
    additional_columns = [model for model in df.columns if model not in MODELS]
    final_columns_order = ordered_columns + additional_columns
    return df[final_columns_order]


tdc_dict = load_results('results/sweep_results_dict.pickle')
tdc_df = save_to_csv(tdc_dict)
y_index = tdc_df.index
# print(tdc_df)
# print(tdc_df.columns)

overlap = list(set(df.columns) & set(tdc_df.columns))
print(f"{overlap=}")

data = df[overlap].combine_first(tdc_df[overlap]).T

print(data)
print(data.index)
print(data.columns)


import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt

pearson_p_value_threshold = 0.05
spearman_p_value_threshold = 0.1
correlation_df_pearson = pd.DataFrame(index=y_index, columns=x_index)
correlation_df_spearman = pd.DataFrame(index=y_index, columns=x_index)

for x in x_index:
    for y in y_index:
        combined_series = pd.concat([data[x], data[y]], axis=1).dropna()

        if len(combined_series) > 5:
            corr_pearson, p_val_pearson = scipy.stats.pearsonr(combined_series.iloc[:, 0], combined_series.iloc[:, 1])
            corr_spearman, p_val_spearman = scipy.stats.spearmanr(combined_series.iloc[:, 0], combined_series.iloc[:, 1])

            correlation_df_pearson.loc[y, x] = corr_pearson if p_val_pearson < pearson_p_value_threshold else 'ns'
            correlation_df_spearman.loc[y, x] = corr_spearman if p_val_spearman < spearman_p_value_threshold else 'ns'
        else:
            correlation_df_pearson.loc[y, x] = 'no data'
            correlation_df_spearman.loc[y, x] = 'no data'


print("Pearson's Correlation:")
print(correlation_df_pearson)
correlation_df_pearson.to_csv("results/pearson_correlation_analysis.csv")

print("\nSpearman's Correlation:")
print(correlation_df_spearman)
correlation_df_spearman.to_csv("results/spearman_correlation_analysis.csv")






x_index_to_plot = ["graph_l1000_vcap/auroc/val", "graph_l1000_vcap/avpr/val", "graph_pcqm4m_g25/mae/val"]

for x in x_index_to_plot:
    task, metric, split = x.split('/')
    num_plots = len(y_index)
    cols = int(num_plots ** 0.5)
    rows = num_plots // cols + (num_plots % cols > 0)
    fig, axs = plt.subplots(rows, cols, figsize=(15, 15))
    fig.suptitle(f"Scatter Plots for {task}_{metric}")

    for i, y in enumerate(y_index):
        combined_series = pd.concat([data[x], data[y]], axis=1).dropna()

        if len(combined_series) > 5:
            ax = axs[i // cols, i % cols]
            ax.scatter(combined_series.iloc[:, 0], combined_series.iloc[:, 1])
            ax.set_title(f"{x}\nvs\n{y}")
            ax.set_xlabel(f"{split}_{metric}")
            ax.set_ylabel(f"{BENCHMARKS[y]}")

            # Display Pearson and Spearman coefficients
            pearson_corr = correlation_df_pearson.loc[y, x]
            spearman_corr = correlation_df_spearman.loc[y, x]
            pearson_info = f"Pearson: {pearson_corr:.2f}" if isinstance(pearson_corr, float) else "Pearson: No sig. corr."
            spearman_info = f"Spearman: {spearman_corr:.2f}" if isinstance(spearman_corr, float) else "Spearman: No sig. corr."
            correlation_info = f"{pearson_info}\n{spearman_info}"
            ax.text(0.05, 0.95, correlation_info, transform=ax.transAxes, verticalalignment='top')
        else:
            axs[i // cols, i % cols].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"results/scatter_plot_{task}_{metric}.png")
    plt.clf()
