import os
import wandb
import yaml
import time

# EDIT HERE
YAML_FILE_PATH = 'finetune_on_fingerprints_config.yaml'
os.environ['SWEEP_MODEL_NAME'] = '10M'
os.environ['SWEEP_FINGERPRINTS_PATH'] = 'ids_to_fingerprint.pt'
os.environ['WANDB_ENTITY'] = 'ogb-lsc-comp'
os.environ['WANDB_PROJECT'] = 'scaling_mol_gnns'

# CONST
TDC_BENCHMARKS = [
    # 'Caco2_Wang',
    # 'Bioavailability_Ma',
    # 'Lipophilicity_AstraZeneca',
    # 'Solubility_AqSolDB',
    # 'HIA_Hou',
    # 'Pgp_Broccatelli',
    # 'BBB_Martins',
    # 'PPBR_AZ',
    # 'VDss_Lombardo',
    # 'CYP2C9_Veith',
    # 'CYP2D6_Veith',
    # 'CYP3A4_Veith',
    # 'CYP2C9_Substrate_CarbonMangels',
    # 'CYP2D6_Substrate_CarbonMangels',
    # 'CYP3A4_Substrate_CarbonMangels',
    # 'Half_Life_Obach',
    'Clearance_Hepatocyte_AZ',
    'Clearance_Microsome_AZ',
    'LD50_Zhu',
    'hERG',
    'AMES',
    'DILI'
]


def create_sweep_and_get_id():
    with open(YAML_FILE_PATH, 'r') as file: sweep_config = yaml.safe_load(file)
    sweep_name = f"{os.getenv('SWEEP_MODEL_NAME')}|{os.getenv('SWEEP_DATASET')}"
    sweep_config['name'] = sweep_name
    return wandb.sweep(sweep=sweep_config, entity=os.getenv('WANDB_ENTITY'), project=os.getenv('WANDB_PROJECT'))

def get_sweep_status(api):
    return api.sweep(f"{os.getenv('WANDB_ENTITY')}/{os.getenv('WANDB_PROJECT')}/{sweep_id}").state

def get_sweep_id_by_name(api, sweep_name):
    sweeps = api.sweeps(path=f"{os.getenv('WANDB_ENTITY')}/{os.getenv('WANDB_PROJECT')}")
    for sweep in sweeps:
        if sweep.name == sweep_name:
            return sweep.id
    return None


if __name__ == "__main__":
    wandb.login()
    api = wandb.Api()

    for dataset in TDC_BENCHMARKS:
        os.environ['SWEEP_DATASET'] = dataset
        sweep_name = f"{os.getenv('SWEEP_MODEL_NAME')}|{dataset}"

        sweep_id = get_sweep_id_by_name(api, sweep_name)
        if sweep_id is not None:
            status = get_sweep_status(api, sweep_id)
            if status == 'FINISHED':
                print(f"Sweep '{sweep_name}' is already finished, moving to the next benchmark.")
                continue
        else:
            sweep_id = create_sweep_and_get_id(api, sweep_name)
            print(f"Created sweep with ID {sweep_id} for dataset {dataset}")

        wandb.agent(sweep_id)
        while get_sweep_status(api, sweep_id) != 'FINISHED':
            time.sleep(100)  # every 100 secs check if sweep finished
