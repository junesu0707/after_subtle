import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

def perform_kruskal_wallis_with_permutation(df, group_column, subcluster_column, num_permutations=1000):
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


def visualize_permutation_results(subcluster, kw_stat, perm_stats):
    plt.figure(figsize=(8, 4))
    plt.hist(perm_stats, bins=30, alpha=0.7, label='Permutation H-stats')
    plt.axvline(x=kw_stat, color='r', linestyle='dashed', linewidth=2, label='Observed H-stat')
    plt.title(f"Permutation Test Visualization for Subcluster {subcluster}")
    plt.xlabel('H-statistic')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    raw_pvalue = np.mean([stat >= kw_stat for stat in perm_stats])
    print(f"Subcluster {subcluster} - Raw P-Value: {raw_pvalue}")