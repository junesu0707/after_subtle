import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage.filters import gaussian_filter

def get_folder_list(directory_path):
    folder_list = []
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isdir(item_path):
            folder_list.append(item_path)
    return folder_list


def divide_groups(dataframe, criteria=[], identity_groups=[]):
    """
    Divide groups based on given criteria or identity groups.
    
    Parameters:
    - dataframe: pandas DataFrame that contains the data
    - criteria: list of column names by which to divide the groups (e.g., ['Gene', 'Gender'])
    - identity_groups: list of lists containing identity keys for each group (e.g., [[1,2,3], [4,5,6]])
    
    Returns:
    - A dictionary containing the divided groups
    """
    
    if criteria and identity_groups:
        print("Please specify either criteria or identity_groups, not both.")
        return
    
    grouped_data = {}
    
    if criteria:
        # Using criteria to divide the groups
        if len(criteria) == 1:
            # If there's only one criterion
            unique_values = dataframe[criteria[0]].unique()
            for value in unique_values:
                grouped_data[value] = dataframe[dataframe[criteria[0]] == value]
        else:
            # If there are multiple criteria
            groups = dataframe.groupby(criteria)
            for name, group in groups:
                grouped_data[name] = group.reset_index(drop=True)
    
    elif identity_groups:
        # Using identity to divide the groups
        for i, group in enumerate(identity_groups):
            grouped_data[f'Group_{i+1}'] = dataframe[dataframe['Identity'].isin(group)]
            
    else:
        print("No criteria or identity groups specified. Returning the original dataframe.")
        grouped_data['Original'] = dataframe
    
    return grouped_data


def density_map(folder_list, cmap = None):
    """Create a density map of the embeddings in the specified directory.
    The directory should contain subdirectories with the embeddings.csv file."""

    # Spatial bin size
    bin_size = 0.1

    # initialize empty lists to hold all x and y values from all files
    all_x = []
    all_y = []

    # loop over all the folders
    for folder in folder_list:
        # check if the folder contains the desired files
        embeddings_file = os.path.join(folder, 'embeddings.csv')

        if os.path.isfile(embeddings_file):
            # read the embeddings and subclusters files into pandas dataframes
            embeddings = pd.read_csv(embeddings_file)

            # assuming the first column is x and the second column is y
            x = embeddings.iloc[:, 0]
            y = embeddings.iloc[:, 1]

            # append the x and y values to the all_x and all_y lists
            all_x.extend(x)
            all_y.extend(y)

    # Compute the number of bins based on the desired bin size
    x_min = min(all_x)
    x_max = max(all_x)
    y_min = min(all_y)
    y_max = max(all_y)
    num_bins_x = int((x_max - x_min) / bin_size)
    num_bins_y = int((y_max - y_min) / bin_size)

    # create a 2D histogram (heatmap) of the point density with the specified number of bins
    heatmap, xedges, yedges = np.histogram2d(all_x, all_y, bins=[num_bins_x, num_bins_y], range=[[x_min, x_max], [y_min, y_max]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    # # apply Gaussian smoothing to the density data
    # heatmap_smoothed = gaussian_filter(heatmap, sigma=sigma)

    # apply logarithmic transformation to the density data
    #heatmap_log = np.log(heatmap + 1000) / np.log(log_base)

    # apply square root transformation
    heatmap_sqrt = np.sqrt(heatmap)

    # normalize the log-transformed heatmap to set zero density to white
    heatmap_normalized = heatmap_sqrt / np.max(heatmap_sqrt)

    # create a colormap with white for zero density and vibrant maximum color
    if cmap is None:
        cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', [(0, 'white'), (1, (0.75, 0, 0, 1.0))])

    # plot the heatmap
    plt.imshow(heatmap_normalized.T, extent=extent, origin='lower', cmap=cmap, aspect='auto')
    plt.colorbar(label='Square Root Density')
    #plt.show()


def density_map_diff(group1_folder_list, group2_folder_list, bin_size=0.15):
    """
    Create a map of density differences between two specified folder lists.
    The lists should contain subdirectories with the embeddings.csv file.
    """

    # initialize empty lists to hold all x and y values from all files for each group
    group1_x = []
    group1_y = []
    group2_x = []
    group2_y = []

    # loop over all the folders in Group 1
    for folder in group1_folder_list:
        # check if the folder contains the desired files
        embeddings_file = os.path.join(folder, 'embeddings.csv')

        if os.path.isfile(embeddings_file):
            # read the embeddings and subclusters files into pandas dataframes
            embeddings = pd.read_csv(embeddings_file)

            # assuming the first column is x and the second column is y
            x = embeddings.iloc[:, 0]
            y = embeddings.iloc[:, 1]

            # append the x and y values to the group1_x and group1_y lists
            group1_x.extend(x)
            group1_y.extend(y)

    # loop over all the folders in Group 2
    for folder in group2_folder_list:
        # check if the folder contains the desired files
        embeddings_file = os.path.join(folder, 'embeddings.csv')

        if os.path.isfile(embeddings_file):
            # read the embeddings and subclusters files into pandas dataframes
            embeddings = pd.read_csv(embeddings_file)

            # assuming the first column is x and the second column is y
            x = embeddings.iloc[:, 0]
            y = embeddings.iloc[:, 1]

            # append the x and y values to the group2_x and group2_y lists
            group2_x.extend(x)
            group2_y.extend(y)

    # Compute the number of bins based on the desired bin size
    x_min = min(min(group1_x), min(group2_x))
    x_max = max(max(group1_x), max(group2_x))
    y_min = min(min(group1_y), min(group2_y))
    y_max = max(max(group1_y), max(group2_y))
    num_bins_x = int((x_max - x_min) / bin_size)
    num_bins_y = int((y_max - y_min) / bin_size)

    # Create the 2D histograms for both groups
    group1_heatmap, xedges, yedges = np.histogram2d(group1_x, group1_y, bins=[num_bins_x, num_bins_y], range=[[x_min, x_max], [y_min, y_max]])
    group2_heatmap, _, _ = np.histogram2d(group2_x, group2_y, bins=[num_bins_x, num_bins_y], range=[[x_min, x_max], [y_min, y_max]])

    # Calculate the density difference map
    dendiff = group1_heatmap / len(group1_x) - group2_heatmap / len(group2_x)
    
    # Apply square root transformation based on the sign of dendiff
    dendiff_sqrt = np.where(dendiff >= 0, np.sqrt(dendiff), -np.sqrt(np.abs(dendiff)))

    # Calculate the maximum density difference
    max_dendiff_sqrt = max(np.max(dendiff_sqrt), -np.min(dendiff_sqrt))
    
    return max_dendiff_sqrt


def draw_dendiff_sqrt(group1_folder_list, group2_folder_list, bin_size=0.15, max_dendiff_sqrt=None, cmap=None):
        # initialize empty lists to hold all x and y values from all files for each group
    group1_x = []
    group1_y = []
    group2_x = []
    group2_y = []

    # loop over all the folders in Group 1
    for folder in group1_folder_list:
        # check if the folder contains the desired files
        embeddings_file = os.path.join(folder, 'embeddings.csv')

        if os.path.isfile(embeddings_file):
            # read the embeddings and subclusters files into pandas dataframes
            embeddings = pd.read_csv(embeddings_file)

            # assuming the first column is x and the second column is y
            x = embeddings.iloc[:, 0]
            y = embeddings.iloc[:, 1]

            # append the x and y values to the group1_x and group1_y lists
            group1_x.extend(x)
            group1_y.extend(y)

    # loop over all the folders in Group 2
    for folder in group2_folder_list:
        # check if the folder contains the desired files
        embeddings_file = os.path.join(folder, 'embeddings.csv')

        if os.path.isfile(embeddings_file):
            # read the embeddings and subclusters files into pandas dataframes
            embeddings = pd.read_csv(embeddings_file)

            # assuming the first column is x and the second column is y
            x = embeddings.iloc[:, 0]
            y = embeddings.iloc[:, 1]

            # append the x and y values to the group2_x and group2_y lists
            group2_x.extend(x)
            group2_y.extend(y)

    # Compute the number of bins based on the desired bin size
    x_min = min(min(group1_x), min(group2_x))
    x_max = max(max(group1_x), max(group2_x))
    y_min = min(min(group1_y), min(group2_y))
    y_max = max(max(group1_y), max(group2_y))
    num_bins_x = int((x_max - x_min) / bin_size)
    num_bins_y = int((y_max - y_min) / bin_size)

    # Create the 2D histograms for both groups
    group1_heatmap, xedges, yedges = np.histogram2d(group1_x, group1_y, bins=[num_bins_x, num_bins_y], range=[[x_min, x_max], [y_min, y_max]])
    group2_heatmap, _, _ = np.histogram2d(group2_x, group2_y, bins=[num_bins_x, num_bins_y], range=[[x_min, x_max], [y_min, y_max]])

    # Calculate the density difference map
    dendiff = group1_heatmap / len(group1_x) - group2_heatmap / len(group2_x)

    # Apply square root transformation based on the sign of dendiff
    dendiff_sqrt = np.where(dendiff >= 0, np.sqrt(dendiff), -np.sqrt(np.abs(dendiff)))

    # Calculate the maximum density difference
    if max_dendiff_sqrt is None:
        max_dendiff_sqrt = max(np.max(dendiff_sqrt), -np.min(dendiff_sqrt))

    # normalize the square root-transformed heatmap to set zero density to white
    dendiff_sqrt_normalized = dendiff_sqrt / max_dendiff_sqrt
    dendiff_sqrt_normalized_smooth = gaussian_filter(dendiff_sqrt_normalized, sigma=0.7)
    
    # Define the colors
    red_color = np.array([(0.75, 0, 0)])
    red_color = red_color.reshape(1, 3)
    complementary_color = mcolors.rgb_to_hsv(red_color)
    complementary_color[0, 0] = (complementary_color[0, 0] + 0.5) % 1.0
    complementary_color = mcolors.hsv_to_rgb(complementary_color)

    # Create a colormap with the defined colors
    if cmap is None:
        cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', [(0, complementary_color), (0.5, 'white'), (1, red_color)])

    # Plot the density difference heatmap
    plt.figure()
    heatmap = plt.imshow(dendiff_sqrt_normalized_smooth.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', aspect='auto', cmap=cmap, vmin=-1, vmax=1)

    # Create a colorbar with ticks
    cbar = plt.colorbar(heatmap, label='Square Root Density Difference')
    cbar.set_ticks([np.min(dendiff_sqrt_normalized), 0, np.max(dendiff_sqrt_normalized)])
    cbar.set_ticklabels(['{:.2f}'.format(np.min(dendiff_sqrt_normalized)), '0', '{:.2f}'.format(np.max(dendiff_sqrt_normalized))])
    # Show or save the figure
    # plt.show()


def compare_cluster_occurrence(group_folder_lists, group_names, subcluster_number, target_groups=None):
    cluster_counts = pd.DataFrame(index=range(subcluster_number), columns=group_names)
    cluster_counts[:] = np.nan
    
    for folder_list, group_name in zip(group_folder_lists, group_names):
        group_cluster_counts = {cluster: 0 for cluster in range(subcluster_number)}
        total_frames = 0
        
        for folder in folder_list:
            # Filter out folders that do not contain the subclusters.csv file
            subcluster_file = os.path.join(folder, 'subclusters.csv')
            if os.path.isfile(subcluster_file):
                df = pd.read_csv(subcluster_file, header=None)
                cluster_counts_series = df.iloc[:, 0].value_counts()
            
                for cluster in range(subcluster_number):
                    count = cluster_counts_series.get(cluster, 0)
                    group_cluster_counts[cluster] += count
                    
                total_frames += len(df)
        
        for cluster, count in group_cluster_counts.items():
            occurrence_rate = count / total_frames
            cluster_counts.at[cluster, group_name] = occurrence_rate
    
    cluster_counts = cluster_counts.fillna(0)  # Convert NaN values to 0

    # Sort the clusters by the occurrence rate of the target group
    if target_groups is not None:
        target_group_cluster_counts = cluster_counts.loc[:, target_groups]
        sorted_clusters = target_group_cluster_counts.sum(axis=1).sort_values(ascending=False).index
        cluster_counts = cluster_counts.reindex(sorted_clusters)

    return cluster_counts


def assign_cluster_ranks(group_folder_lists, group_names, subcluster_number):
    cluster_ranks = pd.DataFrame(index=range(subcluster_number), columns=group_names)
    cluster_ranks[:] = np.nan

    for folder_list, group_name in zip(group_folder_lists, group_names):
        group_cluster_counts = {cluster: 0 for cluster in range(subcluster_number)}
        total_frames = 0

        for folder in folder_list:
            subcluster_file = os.path.join(folder, 'subclusters.csv')
            if os.path.isfile(subcluster_file):
                df = pd.read_csv(subcluster_file, header=None)
                cluster_counts = df.iloc[:, 0].value_counts()

                for cluster in range(subcluster_number):
                    count = cluster_counts.get(cluster, 0)
                    group_cluster_counts[cluster] += count

                total_frames += len(df)

        if total_frames > 0:
            ranks = pd.Series(group_cluster_counts).rank(method='min', ascending=False)
            cluster_ranks[group_name] = ranks

    return cluster_ranks


def plot_cluster_ranks(cluster_ranks):
    plt.imshow(cluster_ranks.values, cmap='hot', aspect='auto')
    plt.xticks(range(len(cluster_ranks.columns)), cluster_ranks.columns, rotation=45, ha='right')
    plt.yticks(range(len(cluster_ranks.index)), cluster_ranks.index)
    plt.xlabel('Group')
    plt.ylabel('Cluster Rank')
    cbar = plt.colorbar()
    cbar.set_label('Rank')
    plt.show()


def calculate_knee_point(data):
    # Calculate the cumulative sum of the sorted data
    cumulative_sum = np.cumsum(data)

    # Normalize the cumulative sum
    normalized_cumulative_sum = cumulative_sum / np.max(cumulative_sum)

    # Calculate the differences between each point and the line connecting the first and last points
    differences = np.abs(normalized_cumulative_sum - (normalized_cumulative_sum[-1] - normalized_cumulative_sum[0]))

    # Find the index of the knee point where the difference is maximum
    knee_point_index = np.argmax(differences)

    return knee_point_index


# def find_consecutive_groups(subclusters, coords_data, target_value, dur_thres=3, pre=10, post=20):
#     """Find consecutive groups with the target value in a pandas DataFrame
#     1. subclusters contains subcluster.
#     2. coords_data contains coords.
#     3. target_value is the subcluster number I want to figure out from coords_data.
#     4. dur_thres is the minimum threshold of consecutive frames."""

#     # Find consecutive groups with the target value
#     groups = subclusters[subclusters[0] == target_value].groupby((subclusters[0] != target_value).cumsum())

#     # Get the start and end indices of each group
#     group_indices = [(group.index[0], group.index[-1]) for _, group in groups]

#     # Extract the rows based on the group indices and diff_threshold
#     appended_coords = pd.DataFrame()
#     sorted_indices = []
#     for group in group_indices:
#         start_index = group[0]
#         end_index = group[1]
#         if (end_index - start_index) > dur_thres:
#             extracted_coords = coords_data.iloc[start_index-pre:end_index+1+post]
#             # Append the extracted coords_data based on group_indices rows
#             appended_coords = pd.concat([appended_coords, extracted_coords])
#             sorted_indices.append([start_index, end_index])

#     # Get the start and end indices of each group
#     group_indices = [(group.index[0], group.index[-1]) for _, group in groups]

#     # Extract the rows based on the group indices and dur_thres
#     appended_coords = pd.DataFrame()
#     sorted_indices = []
#     last_end_index = -1  # 마지막으로 처리한 그룹의 끝 인덱스를 저장할 변수

#     for group in group_indices:
#         start_index = group[0]
#         end_index = group[1]
        
#         # 중복되는 경우를 체크해서 건너뜀
#         if start_index <= last_end_index:
#             continue

#         if (end_index - start_index) > dur_thres:
#             extracted_coords = coords_data.iloc[start_index-pre:end_index+1+post]
#             # Append the extracted coords_data based on group_indices rows
#             appended_coords = pd.concat([appended_coords, extracted_coords])
#             sorted_indices.append([start_index, end_index])
#             last_end_index = end_index  # 끝 인덱스 업데이트

#     return sorted_indices, appended_coords

def find_and_extract_coords(subclusters, coords_data, target_value, dur_thres=3, pre=10, post=20):
    """Find consecutive groups with the target value and extract corresponding coords.
    
    Parameters:
    subclusters (pd.DataFrame): DataFrame containing subcluster information.
    coords_data (pd.DataFrame): DataFrame containing coordinates data.
    target_value (int): Target subcluster number.
    dur_thres (int): Minimum threshold for consecutive frames.
    pre (int): Number of frames to include before the consecutive group.
    post (int): Number of frames to include after the consecutive group.

    Returns:
    pd.DataFrame: Extracted coordinates data corresponding to the consecutive groups.
    """
    # Create a boolean array with the same length as subclusters
    boolean_array = np.zeros(len(subclusters), dtype=bool)

    # Identify the start and end indices of consecutive groups
    groups = subclusters[subclusters[0] == target_value].groupby((subclusters[0] != target_value).cumsum())
    for _, group in groups:
        start_index, end_index = group.index[0], group.index[-1]

        # Mark the consecutive group and the pre/post frames as True
        if (end_index - start_index + 1) > dur_thres:
            boolean_array[max(start_index - pre, 0) : min(end_index + 1 + post, len(subclusters))] = True

    # Extract coords data for the True indices in boolean_array
    extracted_coords = coords_data[boolean_array]

    return extracted_coords
