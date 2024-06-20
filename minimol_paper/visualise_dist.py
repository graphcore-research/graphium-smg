import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

file_path = 'bins.pickle'
def load_bins():
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    return {}
bins = load_bins()

# Prepare data for regression plot
atom_numbers = sorted(bins.keys())
regression_errors = [bins[num]['regression'] for num in atom_numbers]

print(max(atom_numbers))
upper_lims = [i for i in range(0, 110, 20)]

# Prepare data for classification plot
abins = {**{i: [] for i in zip(upper_lims, upper_lims[1:])}, **{(100, 500): []}}
for num in atom_numbers:
    for ranges, _ in abins.items():
        if num >= ranges[0] and num < ranges[1]:
            abins[ranges] += bins[num]['classification']

trues = [sum(abins[num]) for num in abins.keys()]
falses = [len(abins[num]) - trues[i] for i, num in enumerate(abins.keys())]
precision  = [t/(t+f) if t+f>0 else 0 for t, f in zip(trues, falses) ]
total  = [t+f for t, f in zip(trues, falses)]
density  = [100*(t+f)/sum(total) for t, f in zip(trues, falses)]

# print(precision)
# print(density)


# Prepare data regression table
abins = {**{i: [] for i in zip(upper_lims, upper_lims[1:])}, **{(100, 500): []}}
for num in atom_numbers:
    for ranges, _ in abins.items():
        if num >= ranges[0] and num < ranges[1]:
            abins[ranges] += bins[num]['regression']
           
means = [np.mean(abins[num]) for num in abins.keys()]
stds = [np.std(abins[num]) for num in abins.keys()]
total_points = sum([len(abins[num]) for num in abins.keys()])
density = [100*len(abins[num])/total_points for num in abins.keys()]

print(means)
print(stds)
print(density)
