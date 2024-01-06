import os
import pickle
import wandb

# Specify your project and entity (username or team name)
project_name = 'scaling_mol_gnns'
entity_name = 'ogb-lsc-comp'
pickle_path = 'sweep_results_dict.pickle'
csv_path = 'sweep_results_table.csv'

DEFINITION_OF_BETTER = {
    'mae': min,
    'r2': max,
    'spearman': max,
    'auroc': max,
    'avpr': max
}

BENCHMARKS = {
    'Caco2_Wang': 'test_mae',
    'Bioavailability_Ma': 'test_auroc',
    'Lipophilicity_AstraZeneca': 'test_mae',
    'Solubility_AqSolDB': 'test_mae',
    'HIA_Hou': 'test_auroc',
    'Pgp_Broccatelli': 'test_auroc',
    'BBB_Martins': 'test_auroc',
    'PPBR_AZ': 'test_mae',
    'VDss_Lombardo': 'test_spearman',
    'CYP2C9_Veith': 'test_auroc',
    'CYP2D6_Veith': 'test_auroc',
    'CYP3A4_Veith': 'test_auroc',
    'CYP2C9_Substrate_CarbonMangels': 'test_auroc',
    'CYP2D6_Substrate_CarbonMangels': 'test_auroc',
    'CYP3A4_Substrate_CarbonMangels': 'test_auroc',
    'Half_Life_Obach': 'test_spearman',
    'Clearance_Hepatocyte_AZ': 'test_spearman',
    'Clearance_Microsome_AZ': 'test_spearman',
    'LD50_Zhu': 'test_mae',
    'hERG': 'test_auroc',
    'AMES': 'test_auroc',
    'DILI': 'test_auroc',
}

WANDB_STATES = {
    'running': False,
    'crashed': False,
    'finished': True
}



def best_score_for_sweep(sweep):
    abs_best_scores = []
    best_val_loss, best_val_loss_idx = None, None
    metric, def_of_better = None, None
    
    for idx, run in enumerate(sweep.runs):
        
        if WANDB_STATES[run.state] is False:
            continue # skip if crashed or unfinished
        
        if metric is None or def_of_better is None: # dataset cant be extracted from a sweep so get it from a run
            metric = BENCHMARKS[run.config['dataset']]
            def_of_better = DEFINITION_OF_BETTER[metric.split('_')[-1]]

        run_history = run.history()
        if metric in run_history.keys():
            abs_best_scores += [def_of_better(run_history[metric])]
            val_loss = def_of_better(run_history['val_loss'])
            if best_val_loss is None:
                best_val_loss = val_loss
                best_val_loss_idx = idx
            else:
                if best_val_loss < val_loss:
                    best_val_loss = val_loss
                    best_val_loss_idx = idx
    
    # use appropriate reduction for the metric to get the best score in the sweep
    abs_best_score = def_of_better(abs_best_scores) if len(abs_best_scores) else 'NaN'
    if best_val_loss is None:
        fair_best_score = 'NaN'
    else:
        fair_best_score = def_of_better(sweep.runs[best_val_loss_idx].history()[metric])
    return abs_best_score, fair_best_score

def load_results(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    return {}

def save_results(results, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(results, file)

def save_to_csv(results, csv_path):
    import pandas as pd
    data = [{'model_name': key[0], 'dataset': key[1], **value} for key, value in results.items()]

    df = pd.DataFrame(data)
    pivot_df = df.pivot(index='dataset', columns='model_name', values=['abs', 'fair'])
    pivot_df = pivot_df.swaplevel(axis='columns').sort_index(axis='columns')
    pivot_df.columns = ['_'.join(col).strip() for col in pivot_df.columns.values]

    print(pivot_df)
    pivot_df.to_csv(csv_path)


if __name__ == "__main__":

    api = wandb.Api()

    project = api.project(name=project_name, entity=entity_name)

    results = load_results(pickle_path)
    sweeps = project.sweeps()

    # filter
    filtered_sweeps = [sweep for sweep in sweeps if "|" in sweep.name]
    for idx, sweep in enumerate(filtered_sweeps):
        model_name, dataset = sweep.name.split('|')
        print(f"Sweep {idx + 1} / {len(filtered_sweeps)} - {model_name} - {dataset}")
        
        if (model_name, dataset) in results:
            print(f"Combination of ({model_name}, {dataset}) already exists in results. Skipping...")
            continue

        _ = sweep.load(force=True) # this is needed otherwise sweep.runs is an empty list
        if WANDB_STATES[sweep.state.lower()] is False:
            print(f"Sweep state - {sweep.state.lower()} - continuing to the next one")
            continue

        abs_best_score, fair_best_score = best_score_for_sweep(sweep)
        results[(model_name, dataset)] = {"abs": abs_best_score, "fair": fair_best_score}
        print(f"{abs_best_score=}, {fair_best_score=}")

    save_results(results, pickle_path)
    save_to_csv(results, csv_path)
