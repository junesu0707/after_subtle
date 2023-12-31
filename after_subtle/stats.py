import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import bootstrap
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import os


def perform_kruskal_wallis_with_permutation(df, group_column = 'Group', subcluster_column = 'subcluster', num_permutations=1000):
    results = []
    unique_subclusters = df[subcluster_column].unique()

    for subcluster in unique_subclusters:
        subcluster_data = df[df[subcluster_column] == subcluster]
        kw_stat, _ = stats.kruskal(*[group_data['occurrence_rate'].values for name, group_data in subcluster_data.groupby(group_column)])

        perm_stats = []
        for _ in range(num_permutations):
            permuted_data = subcluster_data.copy()
            permuted_data[group_column] = np.random.permutation(permuted_data[group_column])
            perm_stat, _ = stats.kruskal(*[group_data['occurrence_rate'].values for name, group_data in permuted_data.groupby(group_column)])
            perm_stats.append(perm_stat)

        raw_pvalue = np.mean([stat >= kw_stat for stat in perm_stats])

        results.append({'subcluster': subcluster, 'perm_stats': perm_stats, 'kw_stat': kw_stat, 'raw_p_value': raw_pvalue})

    results_df = pd.DataFrame(results)
    return results_df


def apply_fdr_correction(results, alpha=0.05):
    corrected_p_values = multipletests(results['raw_p_value'], alpha=alpha, method='fdr_bh')[1]
    results_fdr = results.copy()
    results_fdr['Corrected P-Value'] = corrected_p_values
    return results_fdr


def visualize_permutation_results(subcluster, kw_stat, perm_stats, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(8, 4))
    plt.hist(perm_stats, bins=30, alpha=0.7, label='Permutation H-stats')
    plt.axvline(x=kw_stat, color='r', linestyle='dashed', linewidth=2, label='Observed H-stat')
    plt.title(f"Permutation Test Visualization for Subcluster {subcluster}")
    plt.xlabel('H-statistic')
    plt.ylabel('Frequency')
    plt.legend()

    # Save the figure
    file_name = f"permutation_test_subcluster_{subcluster}.png"
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path, dpi=300)
    plt.close()  

    raw_pvalue = np.mean([stat >= kw_stat for stat in perm_stats])
    print(f"Subcluster {subcluster} - Raw P-Value: {raw_pvalue}")
    print(f"Figure saved to {save_path}")


# Define a function to calculate bootstrap confidence intervals
def bootstrap_confidence_interval(data, num_iterations=1000, confidence_level=0.95):
    # Calculate mean for bootstrap samples
    means = np.array([np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(num_iterations)])
    lower_bound = np.percentile(means, (1-confidence_level)/2*100)
    upper_bound = np.percentile(means, (1+confidence_level)/2*100)
    return lower_bound, upper_bound
