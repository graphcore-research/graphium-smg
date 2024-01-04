import wandb

# Specify your project and entity (username or team name)
project_name = 'scaling_mol_gnns'
entity_name = 'ogb-lsc-comp'

BENCHMARKS = {
    'Caco2_Wang': {'metric_name': 'test_mae', 'definition_of_better': min},
    'Bioavailability_Ma': {'metric_name': 'test_auroc', 'definition_of_better': max},
    'Lipophilicity_AstraZeneca': {'metric_name': 'test_mae', 'definition_of_better': min},
    'Solubility_AqSolDB': {'metric_name': 'test_mae', 'definition_of_better': min},
    'HIA_Hou': {'metric_name': 'test_auroc', 'definition_of_better': max},
    'Pgp_Broccatelli': {'metric_name': 'test_auroc', 'definition_of_better': max},
    'BBB_Martins': {'metric_name': 'test_auroc', 'definition_of_better': max},
    'PPBR_AZ': {'metric_name': 'test_mae', 'definition_of_better': min},
    'VDss_Lombardo': {'metric_name': 'test_spearman', 'definition_of_better': max},
    'CYP2C9_Veith': {'metric_name': 'test_auroc', 'definition_of_better': max},
    'CYP2D6_Veith': {'metric_name': 'test_auroc', 'definition_of_better': max},
    'CYP3A4_Veith': {'metric_name': 'test_auroc', 'definition_of_better': max},
    'CYP2C9_Substrate_CarbonMangels': {'metric_name': 'test_auroc', 'definition_of_better': max},
    'CYP2D6_Substrate_CarbonMangels': {'metric_name': 'test_auroc', 'definition_of_better': max},
    'CYP3A4_Substrate_CarbonMangels': {'metric_name': 'test_auroc', 'definition_of_better': max},
    'Half_Life_Obach': {'metric_name': 'test_spearman', 'definition_of_better': max},
    'Clearance_Hepatocyte_AZ': {'metric_name': 'test_spearman', 'definition_of_better': max},
    'Clearance_Microsome_AZ': {'metric_name': 'test_spearman', 'definition_of_better': max},
    'LD50_Zhu': {'metric_name': 'test_mae', 'definition_of_better': min},
    'hERG': {'metric_name': 'test_auroc', 'definition_of_better': max},
    'AMES': {'metric_name': 'test_auroc', 'definition_of_better': max},
    'DILI': {'metric_name': 'test_auroc', 'definition_of_better': max}
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
            def_of_better = metric['definition_of_better']

        run_history = run.history()
        if metric['metric_name'] in run_history.keys():
            abs_best_scores += [def_of_better(run_history[metric['metric_name']])]
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
        fair_best_score = def_of_better(sweep.runs[best_val_loss_idx].history()[metric['metric_name']])
    return abs_best_score, fair_best_score


if __name__ == "__main__":

    api = wandb.Api()

    project = api.project(name=project_name, entity=entity_name)

    results = {}
    sweeps = project.sweeps()
    
    for idx, sweep in enumerate(sweeps):
        print(f"Sweep {idx + 1} / {len(sweeps)} - {sweep.name}")
        _ = sweep.load(force=True) # this is needed otherwise sweep.runs is an empty list
        if WANDB_STATES[sweep.state.lower()] is False:
            print(f"Sweep state - {sweep.state.lower()} - continuing to the next one")
            continue
        abs_best_score, fair_best_score = best_score_for_sweep(sweep)
        results[sweep.name] = {"abs": abs_best_score, "fair": fair_best_score}
        print(f"{abs_best_score=}, {fair_best_score=}")

    import json
    print(json.dumps(results, indent=5))
