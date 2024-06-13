import os
import wandb
import yaml
import time
import argparse
import random

# EDIT
#YAML_FILE_PATH = 'finetune_on_fingerprints_config.yaml'

# CONST
TDC_BENCHMARKS = [
    'Caco2_Wang',
    'Bioavailability_Ma',
    'Lipophilicity_AstraZeneca',
    'Solubility_AqSolDB',
    'HIA_Hou',
    'Pgp_Broccatelli',
    'BBB_Martins',
    'PPBR_AZ',
    'VDss_Lombardo',
    'CYP2C9_Veith',
    'CYP2D6_Veith',
    'CYP3A4_Veith',
    'CYP2C9_Substrate_CarbonMangels',
    'CYP2D6_Substrate_CarbonMangels',
    'CYP3A4_Substrate_CarbonMangels',
    'Half_Life_Obach',
    'Clearance_Hepatocyte_AZ',
    'Clearance_Microsome_AZ',
    'LD50_Zhu',
    'hERG',
    'AMES',
    'DILI'
]

OGB_BENCHMARKS = [
    "ogbg-molbace",
    "ogbg-molbbbp",
    "ogbg-molclintox",
    "ogbg-moltox21",
    "ogbg-moltoxcast",
]

def create_sweep_and_get_id(sweep_name, yaml_file_path):
    with open(yaml_file_path, 'r') as file: sweep_config = yaml.safe_load(file)
    sweep_config['name'] = sweep_name
    return wandb.sweep(sweep=sweep_config, entity=os.getenv('WANDB_ENTITY'), project=os.getenv('WANDB_PROJECT'))

def get_sweep_status(api):
    return api.sweep(f"{os.getenv('WANDB_ENTITY')}/{os.getenv('WANDB_PROJECT')}/{sweep_id}").state

def get_sweep_id_by_name(api, sweep_name):
    project = api.project(name=os.getenv('WANDB_PROJECT'), entity=os.getenv('WANDB_ENTITY'))
    sweeps = project.sweeps()
    for sweep in sweeps:
        if sweep.name == sweep_name:
            return sweep.id
    return None
    
# Example: python sweep_checkpoint.py --fingerprints-path ogb-results/ids_to_fingerprint.pt--benchmark ogb --wandb-project biomol-ogb
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sweep Configuration')
    parser.add_argument('--model-name', type=str, default='10M', help='Name of the sweep model')
    parser.add_argument('--fingerprints-path', type=str, default='ogb-results/ids_to_fingerprint.pt', help='Path to the fingerprints file')
    parser.add_argument('--benchmark', type=str, default='ogb', help='Benchmark (tdc or ogb)')
    parser.add_argument('--wandb-entity', type=str, help='W&B entity')
    parser.add_argument('--wandb-project', type=str, help='W&B project')
    args = parser.parse_args()

    os.environ['SWEEP_MODEL_NAME'] = args.model_name
    os.environ['SWEEP_FINGERPRINTS_PATH'] = args.fingerprints_path
    os.environ['WANDB_ENTITY'] = args.wandb_entity
    os.environ['WANDB_PROJECT'] = args.wandb_project
    os.environ['SWEEP_CROSS_VALIDATION_FOLDS'] = str(5)

    if args.benchmark == "tdc":
        yaml_file_path = "finetune_on_fingerprints_config.yaml"
        benchmarks = TDC_BENCHMARKS
    else:
        yaml_file_path = "finetune_on_ogb_config.yaml"
        benchmarks = OGB_BENCHMARKS
    
    api = wandb.Api()
    #random.shuffle(benchmarks)
    for dataset in benchmarks:
        os.environ['SWEEP_DATASET'] = dataset
        sweep_name = f"{os.getenv('SWEEP_MODEL_NAME')}|{dataset}"

        sweep_id = get_sweep_id_by_name(api, sweep_name)
        if sweep_id is not None:
            status = get_sweep_status(api)
            if status == 'FINISHED':
                print(f"Sweep '{sweep_name}' is already finished, moving to the next benchmark.")
                continue
        else:
            sweep_id = create_sweep_and_get_id(sweep_name, yaml_file_path)
            print(f"Created sweep with ID {sweep_id} for dataset {dataset}")

        wandb.agent(sweep_id)
        while get_sweep_status(api) != 'FINISHED':
            time.sleep(100)  # every 100 secs check if sweep finished
