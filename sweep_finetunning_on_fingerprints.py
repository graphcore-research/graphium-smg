import os
import wandb
import yaml

def create_sweep_and_get_id(project_name, yaml_file_path):
    wandb.login()

    with open(yaml_file_path, 'r') as file: sweep_config = yaml.safe_load(file)
    sweep_name = f"fingerprint-finetunning-{os.getenv('SWEEP_MODEL_SIZE')}-{os.getenv('SWEEP_DATASET')}"
    sweep_config['name'] = sweep_name
    sweep_id = wandb.sweep(sweep=sweep_config, project=project_name)

    print(f"Created sweep with ID: {sweep_id}")
    return sweep_id

if __name__ == "__main__":
    os.environ['SWEEP_DATASET'] = 'bbb-martins'
    os.environ['SWEEP_FINGERPRINTS_PATH'] = 'ids_to_fingerprint.pt'
    os.environ['SWEEP_MODEL_SIZE'] = '10M'

    PROJECT_NAME = 'scaling_mol_gnns'
    YAML_FILE_PATH = 'finetune_on_fingerprints.yaml'

    sweep_id = create_sweep_and_get_id(PROJECT_NAME, YAML_FILE_PATH)

    wandb.agent(sweep_id)
