import os
import pickle
import wandb
import pandas as pd

# Specify your project and entity (username or team name)
project_name = 'scale_mol_gnns_fingerprinting'
entity_name = 'ogb-lsc-comp'
pickle_path = 'results/sweep_results_dict.pickle'
csv_path = 'results/sweep_results_table.csv'

DEFINITION_OF_BETTER = {
    'mae': min,
    'r2': max,
    'spearman': max,
    'auroc': max,
    'avpr': max
}

BENCHMARKS = {
    'Caco2_Wang':                       'val_mae',
    'Bioavailability_Ma':               'val_auroc',
    'Lipophilicity_AstraZeneca':        'val_mae',
    'Solubility_AqSolDB':               'val_mae',
    'HIA_Hou':                          'val_auroc',
    'Pgp_Broccatelli':                  'val_auroc',
    'BBB_Martins':                      'val_auroc',
    'PPBR_AZ':                          'val_mae',
    'VDss_Lombardo':                    'val_spearman',
    'CYP2C9_Veith':                     'val_avpr',
    'CYP2D6_Veith':                     'val_avpr',
    'CYP3A4_Veith':                     'val_avpr',
    'CYP2C9_Substrate_CarbonMangels':   'val_avpr',
    'CYP2D6_Substrate_CarbonMangels':   'val_avpr',
    'CYP3A4_Substrate_CarbonMangels':   'val_auroc',
    'Half_Life_Obach':                  'val_spearman',
    'Clearance_Hepatocyte_AZ':          'val_spearman',
    'Clearance_Microsome_AZ':           'val_spearman',
    'LD50_Zhu':                         'val_mae',
    'hERG':                             'val_auroc',
    'AMES':                             'val_auroc',
    'DILI':                             'val_auroc'
}

TEST_BENCHMARKS = {
    'Caco2_Wang':                       'test_ensemble_mae',
    'Bioavailability_Ma':               'test_ensemble_auroc',
    'Lipophilicity_AstraZeneca':        'test_ensemble_mae',
    'Solubility_AqSolDB':               'test_ensemble_mae',
    'HIA_Hou':                          'test_ensemble_auroc',
    'Pgp_Broccatelli':                  'test_ensemble_auroc',
    'BBB_Martins':                      'test_ensemble_auroc',
    'PPBR_AZ':                          'test_ensemble_mae',
    'VDss_Lombardo':                    'test_ensemble_spearman',
    'CYP2C9_Veith':                     'test_ensemble_avpr',
    'CYP2D6_Veith':                     'test_ensemble_avpr',
    'CYP3A4_Veith':                     'test_ensemble_avpr',
    'CYP2C9_Substrate_CarbonMangels':   'test_ensemble_avpr',
    'CYP2D6_Substrate_CarbonMangels':   'test_ensemble_avpr',
    'CYP3A4_Substrate_CarbonMangels':   'test_ensemble_auroc',
    'Half_Life_Obach':                  'test_ensemble_spearman',
    'Clearance_Hepatocyte_AZ':          'test_ensemble_spearman',
    'Clearance_Microsome_AZ':           'test_ensemble_spearman',
    'LD50_Zhu':                         'test_ensemble_mae',
    'hERG':                             'test_ensemble_auroc',
    'AMES':                             'test_ensemble_auroc',
    'DILI':                             'test_ensemble_auroc'
}

WANDB_STATES = {
    'running': False,
    'crashed': False,
    'killed': False,
    'failed': False,
    'finished': True,
}

# if you want to order the columns in the table, specify the order here
MODELS = [ 
    # 'SUPER',
    'ENSEMBLE_GINE_large-sweep_10M',
    # 'ENSEMBLE_GINE_10M', 'ENSEMBLE_GCN_10M', 'ENSEMBLE_10M'
]

def dict_to_bash_command(cmd_dict, script_name="ensemble_eval.py"):
    # Convert each key-value pair in the dictionary to a command line argument
    ignore_list = ['input_dim', 'task_type', 'num_classes', 'wandb_off', 'cpu_slice', 'trainable_params']
    args = [f"--{key.replace('_', '-')}={value}" for key, value in cmd_dict.items() if key not in ignore_list]
    
    # Join all arguments into a single string, prefixed with the script name
    return f"python {script_name} " + " ".join(args)

def find_best_score_for_sweep(sweep):
    mean_val_scores, std_test_scores, run_indices = [], [], []
    metric, def_of_better = None, None
    
    for idx, run in enumerate(sweep.runs):
        
        if WANDB_STATES[run.state] is False:
            continue # skip if crashed or unfinished
        
        if metric is None or def_of_better is None: # dataset cant be extracted from a sweep so get it from a run
            metric, test_metric = BENCHMARKS[run.config['dataset']], TEST_BENCHMARKS[run.config['dataset']]
            def_of_better = DEFINITION_OF_BETTER[metric.split('_')[-1]]

        if "statistics" in run.summary_metrics.keys():
            run_statistics = run.summary_metrics['statistics']
            if f"{metric}" in run_statistics.keys():
                mean_val_scores += [run_statistics[metric]['mean']]
                std_test_scores += [run_statistics[metric]['std']]
                run_indices += [idx]

    # use appropriate reduction for the metric to get the best score in the sweep
    best_mean_test_score = def_of_better(mean_val_scores) if len(mean_val_scores) else 'NaN'

    # Get the index of best_mean_test_score to find the std_test_score
    if best_mean_test_score != 'NaN':
        index_of_best_score = mean_val_scores.index(best_mean_test_score)
        best_mean_test_score = sweep.runs[index_of_best_score].summary_metrics['statistics'][test_metric]['mean']
        best_std_test_score = sweep.runs[index_of_best_score].summary_metrics['statistics'][test_metric]['std']
    else:
        best_std_test_score = 'NaN'
        
    return best_mean_test_score, best_std_test_score

def load_results(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    return {}

def save_results(results, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(results, file)

def save_to_csv(results, csv_path=None):
    # Prepare a list for DataFrame rows
    data = []

    # Iterate through each dataset in BENCHMARKS
    for dataset in BENCHMARKS.keys():
        mean_row = {'Metric': 'Mean', 'Dataset': dataset}
        std_row = {'Metric': 'Std', 'Dataset': dataset}

        # Iterate through results to fill the rows
        for (model_name, result_dataset), values in results.items():
            if dataset == result_dataset:
                mean_row[model_name] = values['mean']
                std_row[model_name] = values['std']

        data.append(mean_row)
        data.append(std_row)

    # Convert list to DataFrame
    df = pd.DataFrame(data)

    # Set the 'Metric' and 'Dataset' columns as a multi-index
    df.set_index(['Metric', 'Dataset'], inplace=True)

    # Handle unspecified order in MODELS or additional columns
    ordered_columns = [model for model in MODELS if model in df.columns]
    additional_columns = [model for model in df.columns if model not in MODELS]
    final_columns_order = ordered_columns + additional_columns

    df = df[final_columns_order]

    if csv_path is not None:
        df.to_csv(csv_path)

    return df



if __name__ == "__main__":

    api = wandb.Api()

    project = api.project(name=project_name, entity=entity_name)

    # results = load_results(pickle_path)
    results = {}

    sweeps = project.sweeps()

    # filter
    filtered_sweeps = [sweep for sweep in sweeps if "|" in sweep.name]

    for idx, sweep in enumerate(filtered_sweeps):
        model_name, dataset = sweep.name.split('|')
        print(f"Sweep {idx + 1} / {len(filtered_sweeps)} - {model_name} - {dataset}")
        
        if model_name not in MODELS:
            print(f"Model {model_name} not selected for analysis. Skipping...")
            continue

        if (model_name, dataset) in results:
            print(f"Combination of ({model_name}, {dataset}) already exists in results. Skipping...")
            continue

        _ = sweep.load(force=True) # this is needed otherwise sweep.runs is an empty list
        if WANDB_STATES[sweep.state.lower()] is False and model_name != 'ENSEMBLE_GINE_large-sweep_10M':
            print(f"Sweep state - {sweep.state.lower()} - continuing to the next one")
            continue

        mean_score, std_score = find_best_score_for_sweep(sweep)
        results[(model_name, dataset)] = {"mean": mean_score, "std": std_score}
        print(f"{mean_score=}, {std_score=}")

    # save_results(results, pickle_path)
    save_to_csv(results, csv_path)
