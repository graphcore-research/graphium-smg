import os

import hydra
import torch
from lightning.pytorch.utilities.model_summary import ModelSummary
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from graphium.config._loader import (
    load_accelerator,
    load_mup,
    load_datamodule,
    get_checkpoint_path,
)
from graphium.trainer.predictor import PredictorModule

from tqdm import tqdm
from copy import deepcopy
import datamol as dm
import sys
from torch_geometric.data import Batch

TESTING_ONLY_CONFIG_KEY = "testing_only"

DATASETS = [
    "ogbg-molbace",
    "ogbg-molbbbp",
    "ogbg-moltox21",
    "ogbg-molclintox",
    "ogbg-moltoxcast",
]
from ogb.graphproppred import PygGraphPropPredDataset
import pandas as pd

@hydra.main(version_base=None, config_path="../../expts/hydra-configs", config_name="main")
def cli(cfg: DictConfig) -> None:
    """
    The main CLI endpoint for training, fine-tuning and evaluating Graphium models.
    """
    return get_final_fingerprints(cfg)


def get_final_fingerprints(cfg: DictConfig) -> None:
    """
    The main (pre-)training and fine-tuning loop.
    """
    # Get ADMET SMILES strings
    if not os.path.exists("saved_ogb_smiles.pt"):
        ogb_mol_ids = set()
        for dataset in DATASETS:
            PygGraphPropPredDataset(root="ogb-data", name=dataset)
            data = pd.read_csv(f"ogb-data/{dataset.replace('-', '_')}/mapping/mol.csv.gz")
            ogb_mol_ids |= set(data["smiles"].apply(dm.unique_id))

        smiles_to_process = []
        ogb_mol_ids_to_find = deepcopy(ogb_mol_ids)

        for dataset in tqdm(DATASETS, desc="Matching molecules to IDs", file=sys.stdout):
            data = pd.read_csv(f"ogb-data/{dataset.replace('-', '_')}/mapping/mol.csv.gz")
            mols = set(data["smiles"])
            for smiles in mols:
                mol_id = dm.unique_id(smiles)
                if mol_id in ogb_mol_ids_to_find:
                    smiles_to_process.append(smiles)
                    ogb_mol_ids_to_find.remove(mol_id)

        assert set(dm.unique_id(s) for s in smiles_to_process) == ogb_mol_ids
        torch.save(smiles_to_process, "saved_ogb_smiles.pt")
    else:
        smiles_to_process = torch.load("saved_ogb_smiles.pt")

    unresolved_cfg = OmegaConf.to_container(cfg, resolve=False)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    ## == Instantiate all required objects from their respective configs ==
    # Accelerator
    cfg, accelerator_type = load_accelerator(cfg)
    assert accelerator_type == "cpu", "get_ogb_fingerprints script only runs on CPU for now"

    ## Data-module
    datamodule = load_datamodule(cfg, accelerator_type)

    # Featurize SMILES strings
    input_features_save_path = "ogb_input_features.pt"
    idx_none_save_path = "ogb_idx_none.pt"
    if not os.path.exists(input_features_save_path):
        input_features, idx_none = datamodule._featurize_molecules(smiles_to_process)

        torch.save(input_features, input_features_save_path)
        torch.save(idx_none, idx_none_save_path)
    else:
        input_features = torch.load(input_features_save_path)

    failures = 0

    # Cast to FP32
    for input_feature in tqdm(input_features, desc="Casting to FP32"):
        try:
            if not isinstance(input_feature, str):
                for k, v in input_feature.items():
                    if isinstance(v, torch.Tensor):
                        if v.dtype == torch.half:
                            input_feature[k] = v.float()
                        elif v.dtype == torch.int32:
                            input_feature[k] = v.long()
            else:
                failures += 1
        except Exception as e:
            print(f"{input_feature = }")
            raise e

    print(f"{failures = }")
                    

    # Load pre-trained model
    predictor = PredictorModule.load_pretrained_model(
        name_or_path=get_checkpoint_path(cfg), device=accelerator_type
    )
    predictor = load_mup(mup_base_path=cfg['architecture']['mup_base_path'], predictor=predictor)

    logger.info(predictor.model)
    logger.info(ModelSummary(predictor, max_depth=4))

    batch_size = 100

    # Run the model to get fingerprints
    results_folder = "ogb-results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    for i, index in tqdm(enumerate(range(0, len(input_features), batch_size))):
        batch = Batch.from_data_list(input_features[index:(index + batch_size)])
        model_fp32 = predictor.model.float()
        output, extras = model_fp32.forward(batch, extra_return_names=["pre_task_heads"])
        fingerprint = extras['pre_task_heads']['graph_feat']
        num_molecules = min(batch_size, fingerprint.shape[0])
        results = [fingerprint[i] for i in range(num_molecules)]

        torch.save(results, f'{results_folder}/res-{i:04}.pt')

        if index == 0:
            print(fingerprint.shape)


    # combine the results
    all_results = []

    for i, index in tqdm(enumerate(range(0, len(input_features), batch_size))):

        results = torch.load(f'{results_folder}/res-{i:04}.pt')
        all_results.extend(results)

    del input_features

    # Save .pt files
    suffix = '_' + unresolved_cfg['run_name_suffix'] if 'run_name_suffix' in unresolved_cfg.keys() else ''
        
    torch.save(all_results, f"{results_folder}/results{suffix}.pt")

    # Generate dictionary SMILES -> fingerprint vector
    smiles_to_fingerprint = dict(zip(smiles_to_process, all_results))
    torch.save(smiles_to_fingerprint, f"{results_folder}/smiles_to_fingerprint{suffix}.pt")

    # Generate dictionary unique IDs -> fingerprint vector
    ids = [dm.unique_id(smiles) for smiles in smiles_to_process]
    ids_to_fingerprint = dict(zip(ids, all_results))
    torch.save(ids_to_fingerprint, f"{results_folder}/ids_to_fingerprint{suffix}.pt")


if __name__ == "__main__":
    cli()
