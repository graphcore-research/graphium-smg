import os
import wandb
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt


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
    'Caco2_Wang':                       'test_mae',
    'Bioavailability_Ma':               'test_auroc',
    'Lipophilicity_AstraZeneca':        'test_mae',
    'Solubility_AqSolDB':               'test_mae',
    'HIA_Hou':                          'test_auroc',
    'Pgp_Broccatelli':                  'test_auroc',
    'BBB_Martins':                      'test_auroc',
    'PPBR_AZ':                          'test_mae',
    'VDss_Lombardo':                    'test_spearman',
    'CYP2C9_Veith':                     'test_auroc',
    'CYP2D6_Veith':                     'test_auroc',
    'CYP3A4_Veith':                     'test_auroc',
    'CYP2C9_Substrate_CarbonMangels':   'test_auroc',
    'CYP2D6_Substrate_CarbonMangels':   'test_auroc',
    'CYP3A4_Substrate_CarbonMangels':   'test_auroc',
    'Half_Life_Obach':                  'test_spearman',
    'Clearance_Hepatocyte_AZ':          'test_spearman',
    'Clearance_Microsome_AZ':           'test_spearman',
    'LD50_Zhu':                         'test_mae',
    'hERG':                             'test_auroc',
    'AMES':                             'test_auroc',
    'DILI':                             'test_auroc'
}

WANDB_STATES = {
    'running': False,
    'crashed': False,
    'finished': True
}

import matplotlib.pyplot as plt
import os


def plot_hparam_distribution(top1_hparams, topn_hparams, save_dir='hparam_plots'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    hparams_keys = top1_hparams.keys()

    for key in hparams_keys:
        top1_values = top1_hparams[key]
        topn_values = topn_hparams[key]

        # Creating subplots
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        fig.suptitle(f'Distribution of {key}')

        # Top 1 Plot
        axes[0].bar(top1_values.keys(), top1_values.values())
        axes[0].set_title('Top 1')
        axes[0].set_xlabel(key)
        axes[0].set_ylabel('Frequency')
        axes[0].tick_params(axis='x', rotation=45)

        # Top N Plot
        axes[1].bar(topn_values.keys(), topn_values.values())
        axes[1].set_title(f'Top {n}')
        axes[1].set_xlabel(key)
        axes[1].set_ylabel('Frequency')
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save the figure
        fig.savefig(os.path.join(save_dir, f'{key}_distribution.png'))
        plt.close(fig)


def initialize_hparams(sweeps):
    """ Initialize hparams dictionaries with all possible parameter options set to zero. """
    all_params = {}
    for sweep in sweeps:
        params = sweep.config['parameters']
        for key, param in params.items():
            all_values = param.get('values', [])
            if key in all_params:
                all_params[key].update(all_values)
            else:
                all_params[key] = set(all_values)
    
    initialized_hparams = {key: Counter({val: 0 for val in values}) for key, values in all_params.items()}
    return initialized_hparams

def get_sweep_parameters(sweep):
    """ Extracts parameters being swept from the sweep configuration. """
    return set(sweep.config['parameters'].keys())

def find_top_n_runs_for_sweep(sweep, n=5):
    swept_params = get_sweep_parameters(sweep)
    runs_data = []

    for run in sweep.runs:
        if not WANDB_STATES[run.state]:
            continue

        metric = BENCHMARKS[run.config['dataset']]
        def_of_better = DEFINITION_OF_BETTER[metric.split('_')[-1]]

        run_statistics = run.summary_metrics['statistics']
        if metric in run_statistics.keys():
            mean_score = run_statistics[metric]['mean']
            # Filter run configuration to include only swept parameters
            filtered_config = {k: v for k, v in run.config.items() if k in swept_params}
            runs_data.append((mean_score, filtered_config))

    # Sort and pick top N
    runs_data.sort(key=lambda x: x[0], reverse=def_of_better is max)
    return runs_data[:n]

def update_hparams(hparams, runs):
    for _, config in runs:
        for key, value in config.items():
            if key in hparams:
                hparams[key][value] += 1
            else:
                hparams[key] = Counter({value: 1})

keywords = ['40M-MPNN-easy-th', '11M-easy-th']  # Replace with your actual keywords

def any_keywords_present(sweep_name, keywords):
    return any(keyword in sweep_name for keyword in keywords)

if __name__ == "__main__":
    api = wandb.Api()
    project = api.project(name=project_name, entity=entity_name)

    n = 5
    sweeps = project.sweeps()

    # Initialize hparams dictionaries
    topn_hparams = initialize_hparams(sweeps)
    top1_hparams = initialize_hparams(sweeps)

    # filter
    filtered_sweeps = [sweep for sweep in sweeps if "|" in sweep.name and any_keywords_present(sweep.name, keywords)]
    for idx, sweep in enumerate(filtered_sweeps):
        model_name, dataset = sweep.name.split('|')
        print(f"Sweep {idx + 1} / {len(filtered_sweeps)} - {model_name} - {dataset}")


        _ = sweep.load(force=True)
        if not WANDB_STATES[sweep.state.lower()]:
            print(f"Sweep state - {sweep.state.lower()} - continuing to the next one")
            continue

        top_n_runs = find_top_n_runs_for_sweep(sweep, n=n)
        update_hparams(topn_hparams, top_n_runs)

        if top_n_runs:
            top_1_run = [top_n_runs[0]]  # Taking the top 1 run
            update_hparams(top1_hparams, top_1_run)
        
        import json      
        print("top1")  
        print(json.dumps(top1_hparams, indent=5))
        print(f"top{n}")
        print(json.dumps(topn_hparams, indent=5))

    plot_hparam_distribution(top1_hparams, topn_hparams)
