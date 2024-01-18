import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import bootstrap
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

import os


def perform_kruskal_wallis_with_permutation(df, cluster_column, num_permutations=1000, random_seed=42):
    """
    Perform Kruskal-Wallis test with permutation to test for differences in occurrence rates between groups within clusters.
    Parameters:
    df: DataFrame containing data to be tested.
    cluster_column: Column containing cluster labels.
    num_permutations: Number of permutations to perform.
    random_seed: Random seed for reproducibility.
    Returns: DataFrame containing results of permutation test.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    group_column = 'Group'
    results = []
    unique_clusters = df[cluster_column].unique()

    for cluster in unique_clusters:
        cluster_data = df[df[cluster_column] == cluster]
        groups = cluster_data.groupby(group_column)

        # Check if all values are identical or all zeros within any group
        if all(len(set(group['occurrence_rate'])) <= 1 for _, group in groups):
            print(f"All values are identical or all zeros in cluster {cluster}. Skipping...")
            continue

        try:
            # Kruskal-Wallis test across groups
            kw_stat, _ = kruskal(*[group['occurrence_rate'].values for name, group in groups])

            # Permutation test
            perm_stats = []
            for _ in range(num_permutations):
                permuted_data = cluster_data.copy()
                permuted_data[group_column] = np.random.permutation(permuted_data[group_column])
                perm_stat, _ = kruskal(*[group['occurrence_rate'].values for name, group in permuted_data.groupby(group_column)])
                perm_stats.append(perm_stat)

            # Calculating raw p-value
            raw_pvalue = np.mean([stat >= kw_stat for stat in perm_stats])

            results.append({cluster_column: cluster, 'perm_stats': perm_stats, 'kw_stat': kw_stat, 'raw_p_value': raw_pvalue})
        
        except ValueError as e:
            print(f"Error in performing Kruskal-Wallis for cluster {cluster}: {e}")
            continue

    results_df = pd.DataFrame(results)
    return results_df


def apply_fdr_correction(results, alpha=0.05):
    corrected_p_values = multipletests(results['raw_p_value'], alpha=alpha, method='fdr_bh')[1]
    results_fdr = results.copy()
    results_fdr['Corrected P-Value'] = corrected_p_values
    return results_fdr


def visualize_permutation_results(cluster, kw_stat, perm_stats, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(8, 4))
    plt.hist(perm_stats, bins=30, alpha=0.7, label='Permutation H-stats')
    plt.axvline(x=kw_stat, color='r', linestyle='dashed', linewidth=2, label='Observed H-stat')
    plt.title(f"Permutation Test Visualization for Cluster {cluster}")
    plt.xlabel('H-statistic')
    plt.ylabel('Frequency')
    plt.legend()

    # Save the figure
    file_name = f"permutation_test_cluster_{cluster}.png"
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path, dpi=300)
    plt.close()  

    raw_pvalue = np.mean([stat >= kw_stat for stat in perm_stats])
    print(f"Cluster {cluster} - Raw P-Value: {raw_pvalue}")
    print(f"Figure saved to {save_path}")


# Define a function to calculate bootstrap confidence intervals
def bootstrap_confidence_interval(data, num_iterations=1000, confidence_level=0.95):
    # Calculate mean for bootstrap samples
    means = np.array([np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(num_iterations)])
    lower_bound = np.percentile(means, (1-confidence_level)/2*100)
    upper_bound = np.percentile(means, (1+confidence_level)/2*100)
    return lower_bound, upper_bound
