import wandb
import argparse
import torch
import json
from torch.utils.data import DataLoader, Dataset
from tdc.benchmark_group import admet_group
from functools import partial
import datamol as dm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score, mean_absolute_error


def train_one_epoch(model, dataloader, loss_fn, optimizer, epoch):
    model.train()
    total_loss = 0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs.float())
        loss = loss_fn(outputs.squeeze(), targets.long())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    loss = total_loss / len(dataloader)
    wandb.log({'epoch': epoch, 'train_loss': loss})
    print(f"Epoch {epoch+1} - Train Loss: {loss}")
    return model


def evaluate(model, dataloader, loss_fn, task_type, evaluation_type, epoch):
    model.eval()
    total_loss = 0
    all_probs = [] 
    all_targets = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs.float())
            loss = loss_fn(outputs, targets.long())
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.tolist())
            all_targets.extend(targets.tolist())

    loss = total_loss / len(dataloader)
    if task_type == 'classification':
        auroc = roc_auc_score(all_targets, all_probs)
        avpr = average_precision_score(all_targets, all_probs)
    else:
        r2 = r2_score(all_targets, all_probs)
        mae = mean_absolute_error(all_targets, all_probs)

    metrics = {
        f'{evaluation_type}_loss': loss,
        f'{evaluation_type}_auroc': auroc if task_type == 'classification' else None,
        f'{evaluation_type}_avpr': avpr if task_type == 'classification' else None,
        f'{evaluation_type}_r2': r2 if task_type != 'classification' else None,
        f'{evaluation_type}_mae': mae if task_type != 'classification' else None,
    }
    wandb.log({**metrics, 'epoch': epoch})
    print(json.dumps(metrics, indent=5))
    print()

class Model(nn.Module):
    def __init__(self, input_dim, depth=3, hidden_dim=512, activation_fn='relu', combine_input='concat', num_classes=None, dropout_rate=0.1, **kwargs):
        super(Model, self).__init__()

        if depth < 2:
            raise ValueError("Depth must be at least 2")

        if depth == 2 and combine_input == 'concat' and hidden_dim != input_dim:
            raise ValueError("When depth is 2 and combine_input is 'concat', hidden_dim must match input_dim")

        self.depth = depth
        self.hidden_dim = hidden_dim
        self.combine_input = combine_input
        self.dropout = nn.Dropout(dropout_rate)
        self.layers = nn.ModuleList()

        # Determine activation function
        if activation_fn == 'relu':
            self.activation_fn = F.relu
        # Add other activation functions if necessary

        # Create layers
        for i in range(depth):
            if self.combine_input == 'concat' and i == depth - 2:
                in_dim = input_dim
                out_dim = input_dim
            elif self.combine_input == 'concat' and i == depth - 1:
                in_dim = input_dim + hidden_dim  # Doubled due to concatenation
                out_dim = num_classes if num_classes is not None else 1
            else:
                in_dim = input_dim if i == 0 else hidden_dim
                out_dim = hidden_dim if i < depth - 1 else (num_classes if num_classes is not None else 1)

            self.layers.append(nn.Linear(in_dim, out_dim))

    def forward(self, x):
        original_x = x
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.depth - 1:
                x = self.activation_fn(x)
                x = self.dropout(x)

            if self.combine_input == 'concat' and i == self.depth - 2:
                x = torch.cat((x, original_x), dim=1)

        return x


class SingleInstancePredictionDataset(Dataset):
    def __init__(self, samples_df, task_type):
        self.samples = samples_df['Drug'].tolist()
        self.targets = samples_df['Y'].tolist()
        if task_type == 'classification':
            self.targets = [float(target) for target in self.targets]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = torch.tensor(self.samples[idx])
        target = torch.tensor(self.targets[idx], dtype=torch.long)
        return sample, target

def match_and_replace_input_column(samples_df, i2v):
    transformed_df = samples_df.copy()
    transformed_df["Drug"] = transformed_df['Drug'].apply(
        lambda s: i2v[dm.unique_id(s)].detach().numpy())
    return transformed_df

def determine_task_type(samples_df):
    if np.issubdtype(samples_df['Y'].dtype, np.integer):
        return 'classification', len(samples_df['Y'].unique())
    else:
        return 'regression', None

def model_summary(model):
    print("Model Summary:")
    print(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {trainable_params / 1e6:.2f} M")
    return trainable_params

def dataloader_factory(benchmark, i2v, args):
    match_and_replace_input_column_partial = partial(match_and_replace_input_column, i2v=i2v)
    
    # Split the samples into train, val and test
    train_samples = match_and_replace_input_column_partial(benchmark['train_val'])
    test_samples = match_and_replace_input_column_partial(benchmark['test'])
    train_samples = train_samples.sample(frac=1, random_state=42).reset_index(drop=True)
    val_size = int(len(train_samples) * args.split)
    val_samples = train_samples[:val_size]
    train_samples = train_samples[val_size:]

    # Create datasets
    train_dataset = SingleInstancePredictionDataset(train_samples, args.task_type)
    val_dataset = SingleInstancePredictionDataset(val_samples, args.task_type)
    test_dataset = SingleInstancePredictionDataset(test_samples, args.task_type)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_dim = train_samples['Drug'].iloc[0].shape[0]

    return train_loader, val_loader, test_loader, input_dim


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--fingerprints-path', type=str, default='ids_to_fingerprint.pt', help='Path to ids_to_fingerprint.pt')
    parser.add_argument('--bench', type=str, default='Caco2_Wang', help='Name of the benchmark from admet_group')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--split', type=float, default=0.1, help='Ratio of validation set split')
    parser.add_argument('--depth', type=int, default=3, help='Depth of the model')
    # Model arch
    parser.add_argument('--hidden-dim', type=int, default=512, help='Dimension of hidden layers')
    parser.add_argument('--activation-fn', type=str, default='relu', choices=['relu'], help='Activation function')
    parser.add_argument('--combine-input', type=str, default='concat', choices=['concat', 'other_option'], help='Method to combine input')
    parser.add_argument('--dropout-rate', type=float, default=0.1, help='Dropout rate')

    args = parser.parse_args()

    # Load the id to fingerprint mapping
    i2v = torch.load(args.fingerprints_path)

    # Get the TDC data
    group = admet_group(path='data/')
    benchmark = group.get(args.bench)

    # Determine task type and number of classes if classification
    args.task_type, args.num_classes = determine_task_type(benchmark['train_val'])

    # Construct dataloaders
    train_dl, val_dl, test_dl, args.input_dim = dataloader_factory(benchmark, i2v, args)    

    # Define a model
    model = Model(**vars(args))
    args.trainable_params = model_summary(model)

    # Initialize wandb
    wandb.init(project='scaling_mol_gnns', entity='graphcore', name='10M-ipu_fingerprints_bbb-martins')
    wandb.config.update(args)

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss() if args.task_type == 'classification' else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Test random model
    epoch = 0
    evaluate(model, test_dl, loss_fn, args.task_type, evaluation_type='test', epoch=epoch)

    # Training and validation loop
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        model = train_one_epoch(model, train_dl, loss_fn, optimizer, epoch)
        evaluate(model, val_dl, loss_fn, args.task_type, evaluation_type='val', epoch=epoch)

    # Test trained model
    evaluate(model, test_dl, loss_fn, args.task_type, evaluation_type='test', epoch=epoch)
    wandb.finish()

if __name__ == "__main__":
    main()
