import sys
sys.path.append("..")
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import torch
import os
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold, train_test_split
import csv
import re
import pandas as pd
import seaborn as sns


jobss = []
file1 = open(os.path.join("/data/users4/ziqbal5/ILR", 'output2.txt'), 'r+')
lines = file1.readlines()
lines = [line.replace(' ', '') for line in lines]

start = '/data/users4/ziqbal5/ILR/Data/'

for line in lines:
    jobss.append(str(line.rstrip('\n')))
jobss.sort()
print(jobss)
#print(len(jobss))
n_col = len(jobss) #4  # (*2)
Ad = []

found_jobs = set()

for i in range(n_col):
    job_id = jobss[i]
    found = False
    for dirpath, dirnames, filenames in os.walk(start):
        for dirname in dirnames:
            if dirname.startswith(jobss[i]):
                filename = os.path.join(dirpath, dirname)
                Ad.append(filename)
                found_jobs.add(job_id)
                found = True
                break
            if found:
                break
missing_jobs = [job_id for job_id in jobss if job_id not in found_jobs]
print(f"\n\nMissing job IDs: {missing_jobs}")

#Extract dataset name
first_path = Ad[0]
match = re.search(r"LSTM_(.*?)_ws", first_path)
if match:
    dataset_name = match.group(1)
    print("Extracted dataset name:", dataset_name)
else:
    print("No match found.")



brain_functional_networks_mapping = {
        "sub-cortical": [0,1,2,3,4],
        "auditory": [5,6],
        "sensorimotor": list(range(7,16)),
        "visual": list(range(16,25)),
        "cognitive control": list(range(25,42)),
        "default mode": list(range(42,49)),
        "cerebellar": [49,50,51,52]
    }




combined_y_list = []
combined_latent_list = []

for path_a in Ad:
    # Load combined_y
    with open(os.path.join(path_a, "combined_y.pkl"), "rb") as infile:
        combined_y = pickle.load(infile)  # Shape: [31]
        combined_y_list.append(combined_y)
   

    # Load combined_latent
    with open(os.path.join(path_a, "combined_latent.pkl"), "rb") as infile:
        combined_latent = pickle.load(infile)  # Shape: [31, 200]
        combined_latent_list.append(combined_latent)


# Concatenate all tensors along the first dimension
final_combined_y = torch.cat(combined_y_list, dim=0)  # Shape: [31 * len(Ad)]
final_combined_latent = torch.cat(combined_latent_list, dim=0)  # Shape: [31 * len(Ad), 200]


#classification auc scores for all the jobs.
Auc_dstream = []



for path_a in Ad:


    file_path = os.path.join(path_a, 'ACC_AUC.csv')
    with open(file_path, newline = '') as csvfile:
        a = csv.reader(csvfile, delimiter=',')
        for row in a:
            pass
    row[1] = re.sub(r'[\[\]]', '', str(row[1])) #Accuracy
    #print(row[1])
    row[2] = re.sub(r'[\[\]]', '', str(row[2]))  #AUC
    Auc_dstream.append(float(row[2]))
    #print(row[2])

print("\nAUC downstream task: ", Auc_dstream, len(Auc_dstream))


# Initialize the combined contributions dictionary
combined_region_contributions = {}

#Method 1: take common indices
# # Iterate over each fold's path
# for i, path_a in enumerate(Ad):
#     with open(os.path.join(path_a, "region_contributions.pkl"), "rb") as infile:
#         region_contributions = pickle.load(infile)  # Load dictionary
#     # If first fold, initialize with sets from the first dictionary
#     if i == 0:
#         combined_region_contributions = {k: set(v) for k, v in region_contributions.items()}
#     else:
#         # Perform intersection to retain only common indices across folds
#         for network in combined_region_contributions.keys():
#             if network in region_contributions:
#                 combined_region_contributions[network] &= set(region_contributions[network])  # Intersection


## Convert sets back to sorted lists
#final_region_contributions = {k: sorted(list(v)) for k, v in combined_region_contributions.items()}


#Method 2: take top k most frequent indices
# # Step 1: Initialize a counter to track index frequencies across folds
# index_counts = defaultdict(lambda: defaultdict(int))

# # Step 2: Aggregate counts for each index in each network
# for path_a in Ad:
#     try:
#         with open(os.path.join(path_a, "region_contributions.pkl"), "rb") as infile:
#             region_contributions = pickle.load(infile)
#         for network, indices in region_contributions.items():
#             for idx in indices:
#                 index_counts[network][idx] += 1
#     except FileNotFoundError:
#         print(f"Warning: File not found in {path_a}. Skipping.")
#         continue

# # Step 3: Select top-K most frequent indices per network
# top_k = 100  # Adjust this value based on your needs
# final_region_contributions = {
#     network: sorted([idx for idx, _ in sorted(counts.items(), key=lambda x: -x[1])[:top_k]])
#     for network, counts in index_counts.items()
# }

#Method 3: assign each index to only one network based on frequency.
# --- Step 1: Aggregate index frequencies per network ---
index_counts = defaultdict(lambda: defaultdict(int))
for path_a in Ad:
    try:
        with open(os.path.join(path_a, "region_contributions.pkl"), "rb") as infile:
            region_contributions = pickle.load(infile)
        for network, indices in region_contributions.items():
            for idx in indices:
                index_counts[network][idx] += 1
    except FileNotFoundError:
        print(f"Warning: File not found in {path_a}. Skipping.")
        continue

# --- Step 2: Assign each index to its most frequent network ---
index_to_network = {}
conflict_indices = set()  # Track indices with ties

# First pass: Find the network with max frequency for each index
all_indices = set()
for network, counts in index_counts.items():
    all_indices.update(counts.keys())

for idx in all_indices:
    max_freq = -1
    best_networks = []
    
    for network, counts in index_counts.items():
        if idx in counts:
            freq = counts[idx]
            if freq > max_freq:
                max_freq = freq
                best_networks = [network]
            elif freq == max_freq:
                best_networks.append(network)
    
    if len(best_networks) == 1:
        index_to_network[idx] = best_networks[0]
    else:
        conflict_indices.add(idx)  # Handle ties later

# --- Step 3: Resolve conflicts (optional: assign to first network alphabetically) ---
for idx in conflict_indices:
    # Example: Assign to the network that appears first alphabetically
    networks = []
    for network, counts in index_counts.items():
        if idx in counts:
            networks.append(network)
    index_to_network[idx] = sorted(networks)[0]  # Simple tiebreaker

# --- Step 4: Build final networks with unique indices ---
final_region_contributions = defaultdict(list)
for idx, network in index_to_network.items():
    final_region_contributions[network].append(idx)

# Sort indices for readability
final_region_contributions = {
    network: sorted(indices)
    for network, indices in final_region_contributions.items()
}



print("Final combined y shape:", final_combined_y.shape)
print("Final combined latent shape:", final_combined_latent.shape)

print("Final combined region:",final_region_contributions)


#######################################################################################
###############################Start: T_Test###########################################
#######################################################################################

# Perform one-sample t-tests
t_stats, p_values = ttest_1samp(final_combined_latent.cpu().numpy(), 0, axis=0)  # Compare to baseline = 0

# Correct for multiple comparisons
_, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')

# Identify significant features
significant_features = p_corrected < 0.05
print(f"Number of significant latent features for one sample ttest: {np.sum(significant_features)}")
#############


indices_ones = torch.nonzero(final_combined_y == 1, as_tuple=True)[0]
indices_zeros = torch.nonzero(final_combined_y == 0, as_tuple=True)[0]

X_schiz = final_combined_latent[indices_ones]

X_control = final_combined_latent[indices_zeros]

# Perform two-sample t-tests
t_stats, p_values = ttest_ind(X_schiz.cpu().numpy(), X_control.cpu().numpy(), axis=0)

# Correct for multiple comparisons
_, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

# Compute Cohen’s d for all features
d_values = [cohens_d(X_schiz.cpu().numpy()[:, i], X_control.cpu().numpy()[:, i]) for i in range(X_schiz.shape[1])]


# Identify significant features
significant_features = p_corrected < 0.05

print(f"Number of significant latent features for two sample ttest: {np.sum(significant_features)}")



# # Check directionality (which group has higher/lower means)
# schiz_higher = np.mean(X_schiz.cpu().numpy(), axis=0) > np.mean(X_control.cpu().numpy(), axis=0)

# control_higher = ~schiz_higher

# print(f"Features significantly higher in schizophrenia: {np.sum(significant_features & schiz_higher)}")
# print(f"Features significantly higher in controls: {np.sum(significant_features & control_higher)}")

# indices_sz = np.where(significant_features & schiz_higher)[0]
# indices_hc = np.where(significant_features & control_higher)[0]


# # Get indices of significant features
# significant_indices = np.where(significant_features)[0]

# # Extract significant d-values
# #significant_d_values = [d_values[i] for i in significant_indices]


# region_d_values2 = defaultdict(list)

# for region, indices in final_region_contributions.items():
#     for idx in indices:
#         if idx in significant_indices:
#             region_d_values2[region].append(d_values[idx])



# for network, count in region_d_values2.items():
#     print(f"{network}: {count}")

# #network_sums = {region: sum(values) for region, values in region_d_values2.items()}
# network_averages = {region: np.mean(values) if values else 0 for region, values in region_d_values2.items()}
# # Print the summed d_values per network
# # for region, total in network_sums.items():
# #     print(f"{region}: {total:.4f}")

# for region, total in network_averages.items():
#     print(f"{region}: {total:.4f}")

# # Feature names (including 'auditory', even though it's not in the data)
# feature_names = ['auditory', 'cognitive control', 'default mode', 'cerebellar', 
#                  'sub-cortical', 'visual', 'sensorimotor']

# # Get values in the correct order, assigning zero if missing
# ordered_values = [network_averages.get(feature, 0) for feature in feature_names]

# # Class labels
# class_labels = ['Schizophrenia patients', 'Healthy controls']

# # Plot
# plt.figure(figsize=(10, 6))
# bars = plt.bar(feature_names, ordered_values, 
#                color=['steelblue' if val > 0 else 'coral' for val in ordered_values], 
#                edgecolor='black')

# # Formatting
# plt.axhline(0, color='black', linewidth=0.8, linestyle='--', label='Decision boundary (0)')
# plt.xlabel('Functional Networks')
# plt.ylabel("Averaged Cohen's d Values")
# plt.title('Feature Importance Across Functional Networks')

# # Legend
# handles = [
#     plt.Line2D([0], [0], color='steelblue', lw=4, label=f'Positive: {class_labels[0]}'),
#     plt.Line2D([0], [0], color='coral', lw=4, label=f'Negative: {class_labels[1]}')
# ]
# plt.legend(handles=handles, loc='best')

# # Grid and layout
# plt.grid(axis='y', linestyle='--', alpha=0.6)
# plt.tight_layout()

# # Save the plot
# plt.savefig(Ad[0] + '/ttest_all_folds.png')  # Save in the first directory



# sz_network_counts = defaultdict(int)
# hc_network_counts = defaultdict(int)

# for region, values in region_d_values2.items():
#     sz_network_counts[region] = sum(1 for v in values if v > 0)  # Count positive values
#     hc_network_counts[region] = sum(1 for v in values if v < 0)  # Count negative values

# print(region_d_values2)
# print("Schizophrenia-related counts:", sz_network_counts)
# print("Healthy control-related counts:", hc_network_counts)

# functional_networks = ['auditory', 'cognitive control', 'default mode', 'cerebellar', 
#                         'sub-cortical', 'visual', 'sensorimotor']

# # Get values for each network (default to 0 if not present in the dict)
# sz_values = [sz_network_counts[fn] for fn in functional_networks]
# hc_values = [hc_network_counts[fn] for fn in functional_networks]


# # Bar plot settings
# x = np.arange(len(functional_networks))
# width = 0.35  # Width of bars

# fig, ax = plt.subplots(figsize=(10, 5))
# bars1 = ax.bar(x - width/2, sz_values, width, label='Schizophrenia', color='red')
# bars2 = ax.bar(x + width/2, hc_values, width, label='Healthy Control', color='blue')

# # Labels and title
# ax.set_xlabel('Functional Networks')
# ax.set_ylabel('Number of Significant Features')
# ax.set_title('Comparison of Significant Features Across Functional Networks')
# ax.set_xticks(x)
# ax.set_xticklabels(functional_networks, rotation=45, ha='right')
# ax.legend()

# # Display the plot
# plt.tight_layout()
# plt.savefig(Ad[0] + '/ttest_sig_features.png')





#######################################################################################
###############################End: T_Test#############################################
#######################################################################################


# Initialize lists to collect results across 5 folds
all_ordered_scores = []
all_auc_probes = []
all_accuracy_probes = []

kf = KFold(n_splits=5, shuffle=False)

for train_idx, test_idx in kf.split(final_combined_latent):
    # Split data into training and test sets
    X_train, X_test = final_combined_latent[train_idx].cpu(), final_combined_latent[test_idx].cpu()
    y_train, y_test = final_combined_y[train_idx].cpu(), final_combined_y[test_idx].cpu()

    # Train Logistic Regression model
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)

    # Evaluate model performance on test set
    y_pred = clf.predict(X_test)
    accuracy_probe = accuracy_score(y_test, y_pred)
    auc_probe = roc_auc_score(y_test, y_pred)

    # Store AUC and Accuracy for this fold
    all_auc_probes.append(auc_probe)
    all_accuracy_probes.append(accuracy_probe)

    # Compute feature importance for each region
    coefs = clf.coef_[0]
    group_importance = {}
    group_direction = {}

    for group, indices in final_region_contributions.items():
 
        if not indices:  # Check if indices list is empty
             group_importance[group] = 0
             group_direction[group] = 0
             continue
        group_weight = coefs[indices]
        l2_norm = np.linalg.norm(group_weight)
        mean_weight = np.mean(group_weight) if len(group_weight) > 0 else 0

        group_importance[group] = l2_norm
        group_direction[group] = mean_weight

    # Compute ordered scores based on predefined network order
    predefined_order = ['auditory', 'cognitive control', 'default mode', 'cerebellar', 
                        'sub-cortical', 'visual', 'sensorimotor']

    network_score_dict = dict(zip(group_direction.keys(), group_direction.values()))
    ordered_networks = [net for net in predefined_order if net in network_score_dict]
    ordered_scores = [network_score_dict[net] for net in ordered_networks]

    # Replace NaN values with 0
    ordered_scores = [0 if np.isnan(item) else item for item in ordered_scores]

    # Store ordered scores for this fold
    all_ordered_scores.append(ordered_scores)

# Final collected results after 5 folds
#print("\nCollected Ordered Scores Across Folds:", all_ordered_scores)
print("\nCollected AUC Scores Across Folds (Probe): ", all_auc_probes)
#print("Collected Accuracy Scores Across Folds (Probe): ", all_accuracy_probes)

coefficients = np.array(all_ordered_scores)



# Normalize AUC scores to use them as weights (summing to 1)
weights = all_auc_probes / np.sum(all_auc_probes)
#print(weights)
# Weighted average of coefficients
weighted_mean_coefficients = np.average(coefficients, axis=0, weights=weights)
#print(coefficients)
print('weighted_mean_coefficients: ', weighted_mean_coefficients)
mean_coefficients = np.average(coefficients, axis=0)
print('mean_coefficients :',mean_coefficients)


# Standard deviation remains unweighted to show variability across folds
std_coefficients = np.std(coefficients, axis=0)
print("std_coefficients", std_coefficients)

feature_names = ['auditory', 'cognitive control', 'default mode', 'cerebellar', 
                    'sub-cortical', 'visual', 'sensorimotor']

# Define class labels
class_labels = ['Patients', 'Healthy controls']

# Plotting the mean coefficients with error bars (standard deviation)
plt.figure(figsize=(10, 6))
bars = plt.bar(feature_names, weighted_mean_coefficients, yerr=std_coefficients, capsize=5, 
               color=['steelblue' if coef > 0 else 'coral' for coef in weighted_mean_coefficients], 
               edgecolor='black')

# Add labels and formatting
plt.axhline(0, color='black', linewidth=0.8, linestyle='--', label='Decision boundary (0)')
plt.xlabel('Functional Networks')
plt.ylabel('Weighted Mean Coefficient Value (Importance)')
#plt.title('Weighted Feature Importance with Variability Across Folds (AUC-based)')
plt.title(f'{dataset_name} - Weighted Feature Importance with Variability Across Folds (AUC-based)')
# Add legend for class labels
handles = [
    plt.Line2D([0], [0], color='coral', lw=4, label=f'Positive: {class_labels[0]}'),
    plt.Line2D([0], [0], color='steelblue', lw=4, label=f'Negative: {class_labels[1]}')
]
plt.legend(handles=handles, loc='best')

# Add grid
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Show plot
plt.tight_layout()
#plt.show()
plt.savefig(Ad[0] + '/lr_fc_allfolds.png')







#box whisker plot for ptr overall probe
def transform_auc_list_to_dict(auc_flat_list):

    # Input validation
    if not isinstance(auc_flat_list, list):
        raise TypeError("Input must be a list")
        
    if not auc_flat_list:
        raise ValueError("Input list cannot be empty")
        
    
    # Truncate extra elements if length is not divisible by 10
    if len(auc_flat_list) % 20 != 0:
        valid_length = len(auc_flat_list) - (len(auc_flat_list) % 20)
        truncated_list = auc_flat_list[:valid_length]

        auc_flat_list = truncated_list
        print("\n trancating list because some jobs were missing (not divisible by 20). New length is:  ", len(auc_flat_list))

    if len(auc_flat_list) % 20 != 0:
        raise ValueError(
            f"Input list length ({len(auc_flat_list)}) must be divisible by 5. "
            f"Found {len(auc_flat_list) % 20} extra values."
        )
        
    if not all(isinstance(x, (int, float)) for x in auc_flat_list):
        raise ValueError("All list elements must be numeric")
    
    # Core transformation
    n_methods = len(auc_flat_list) // 20
    return {
        f'ptr {i+1}': auc_flat_list[i*20 : (i+1)*20]
        for i in range(n_methods)
    }

auc_scores = transform_auc_list_to_dict(Auc_dstream)

print(auc_scores)
exit()


# Prepare data
df = pd.DataFrame(auc_scores).melt(var_name='Method', value_name='AUC')
df_overall = pd.DataFrame({'Method': ['Overall'], 'AUC': [df['AUC'].values]})
df_probe = pd.DataFrame({'Method': ['Probe'], 'AUC': [all_auc_probes]})


# Combine all data
plot_data = pd.concat([
    df.assign(Group='Pre-training Methods'),
    df_overall.explode('AUC').assign(Group='Overall'),
    df_probe.explode('AUC').assign(Group='Probe')
])

# Plot
plt.figure(figsize=(18, 6))
sns.set_style("whitegrid")

# Boxplot with custom order
order = list(auc_scores.keys()) + ['Overall', 'Probe']
sns.boxplot(data=plot_data, x='Method', y='AUC', hue='Group', 
            palette={'Pre-training Methods': 'lightblue', 'Overall': 'salmon', 'Probe': 'lightgreen'},
            order=order, dodge=False)

# Add individual points
sns.stripplot(data=plot_data, x='Method', y='AUC', color='black', alpha=0.5, size=4)

# Add mean lines
for i, method in enumerate(order):
    subset = plot_data[plot_data['Method'] == method]
    mean = subset['AUC'].mean()
    plt.hlines(mean, xmin=i-0.4, xmax=i+0.4, colors='red', linestyles='solid', lw=2)

# Formatting
plt.xticks(rotation=45)
plt.ylabel('AUC Score')
plt.title('AUC Distribution: Pre-training Methods vs Overall vs Probe', pad=20)
plt.legend(handles=[
    plt.Line2D([], [], color='red', lw=2, label='Mean'),
    plt.Line2D([], [], color='lightblue', marker='s', markersize=10, linewidth=0, label='Pre-training'),
    plt.Line2D([], [], color='salmon', marker='s', markersize=10, linewidth=0, label='Overall'),
    plt.Line2D([], [], color='lightgreen', marker='s', markersize=10, linewidth=0, label='Probe')
], loc='upper right')
plt.ylim(0.4, 1.0)
plt.savefig(Ad[0] + '/auc_plots.png')

