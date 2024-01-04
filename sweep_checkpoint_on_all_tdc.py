import os
import wandb
import yaml
import time

# Editables
YAML_FILE_PATH = 'finetune_on_fingerprints_config.yaml'
os.environ['SWEEP_MODEL_SIZE'] = '10M'
os.environ['SWEEP_FINGERPRINTS_PATH'] = 'ids_to_fingerprint.pt'
os.environ['WANDB_ENTITY'] = 'ogb-lsc-comp'
os.environ['WANDB_PROJECT'] = 'scaling_mol_gnns'

# Constants
TDC_BENCHMARKS = [
    {'dataset': 'Caco2_Wang', 'metric': 'MAE'},
    {'dataset': 'Bioavailability_Ma', 'metric': 'AUROC'},
    {'dataset': 'Lipophilicity_AstraZeneca', 'metric': 'MAE'},
    {'dataset': 'Solubility_AqSolDB', 'metric': 'MAE'},
    {'dataset': 'HIA_Hou', 'metric': 'AUROC'},
    {'dataset': 'Pgp_Broccatelli', 'metric': 'AUROC'},
    {'dataset': 'BBB_Martins', 'metric': 'AUROC'},
    {'dataset': 'PPBR_AZ', 'metric': 'MAE'},
    {'dataset': 'VDss_Lombardo', 'metric': 'Spearman'},
    {'dataset': 'CYP2C9_Veith', 'metric': 'AUPRC'},
    {'dataset': 'CYP2D6_Veith', 'metric': 'AUPRC'},
    {'dataset': 'CYP3A4_Veith', 'metric': 'AUPRC'},
    {'dataset': 'CYP2C9_Substrate_CarbonMangels', 'metric': 'AUPRC'},
    {'dataset': 'CYP2D6_Substrate_CarbonMangels', 'metric': 'AUPRC'},
    {'dataset': 'CYP3A4_Substrate_CarbonMangels', 'metric': 'AUROC'},
    {'dataset': 'Half_Life_Obach', 'metric': 'Spearman'},
    {'dataset': 'Clearance_Hepatocyte_AZ', 'metric': 'Spearman'},
    {'dataset': 'Clearance_Microsome_AZ', 'metric': 'Spearman'},
    {'dataset': 'LD50_Zhu', 'metric': 'MAE'},
    {'dataset': 'hERG', 'metric': 'AUROC'},
    {'dataset': 'AMES', 'metric': 'AUROC'},
    {'dataset': 'DILI', 'metric': 'AUROC'}
]

def create_sweep_and_get_id():
    with open(YAML_FILE_PATH, 'r') as file: sweep_config = yaml.safe_load(file)
    sweep_name = f"[fingerprint]{os.getenv('SWEEP_MODEL_SIZE')}-{os.getenv('SWEEP_DATASET')}"
    sweep_config['name'] = sweep_name
    return wandb.sweep(sweep=sweep_config, entity=os.getenv('WANDB_ENTITY'), project=os.getenv('WANDB_PROJECT'))

def get_sweep_status(api):
    return api.sweep(f"{os.getenv('WANDB_ENTITY')}/{os.getenv('WANDB_PROJECT')}/{sweep_id}").state

if __name__ == "__main__":
    wandb.login()
    api = wandb.Api()

    for benchmark in TDC_BENCHMARKS:
        os.environ['SWEEP_DATASET'] = benchmark['dataset']

        sweep_id = create_sweep_and_get_id()
        print(f"Created sweep with ID {sweep_id} for dataset {benchmark['dataset']}")

        wandb.agent(sweep_id)
        while get_sweep_status(api) != 'FINISHED':
            time.sleep(100) # every 100 secs check if sweep finished
