import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage.filters import gaussian_filter


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


def density_map(folder_list, range=None, cmap = None):
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
    if range is not None:
        x_min, x_max, y_min, y_max = range
    else:
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
    """Compare the occurrence rate of each cluster in the specified groups.
    The folder lists should contain subdirectories with the subclusters.csv file.
    The target groups should be specified as a list of group names.
    If the target groups are not specified, the occurrence rate of all groups will be compared."""

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


def calculate_filewise_cluster_occurrence(folder_list, cluster_num, cluster):
    """
    Calculate the occurrence rate(occupied frames/total frames) of each cluster in each file within the provided folder list.

    Parameters:
    folder_list (list): List of folders containing 'subclusters.csv' or 'superclusters.csv files.
    cluster_number (int): Number of clusters.
    cluster (str): 'subcluster' or 'supercluster'.

    Returns:
    pd.DataFrame: DataFrame with file names as columns and subcluster numbers as rows.
    """
    filewise_cluster_counts = pd.DataFrame(index=range(cluster_num))
    if cluster == 'subcluster':
        cluster_name = 'subclusters.csv'
    elif cluster == 'supercluster':
        cluster_name = 'superclusters.csv'


    for folder in folder_list:
        cluster_file = os.path.join(folder, cluster_name)
        if os.path.isfile(cluster_file):
            df = pd.read_csv(cluster_file, header=None)

            # Select the specific column for superclusters if cluster_num is provided
            if cluster == 'supercluster' and cluster_num is not None:
                if 0 < cluster_num <= df.shape[1]:
                    df = df.iloc[:, cluster_num-1]  # Adjust column index as needed
                else:
                    raise ValueError(f"Invalid cluster_num: {cluster_num}. It should be within the range of 1 to {df.shape[1]} for supercluster.")
                
            # Handle Series or DataFrame for value_counts
            if isinstance(df, pd.Series):
                cluster_counts_series = df.value_counts()  # Directly use the Series
            else:
                cluster_counts_series = df.iloc[:, 0].value_counts()  # Use the first column if DataFrame

            total_frames = len(df)

            # Calculate occurrence rate for each cluster
            occurrence_rates = {cluster: cluster_counts_series.get(cluster, 0) / total_frames for cluster in range(cluster_num)}
            
            # Add to the DataFrame
            file_name = os.path.basename(folder)
            filewise_cluster_counts[file_name] = filewise_cluster_counts.index.map(occurrence_rates)

    return filewise_cluster_counts.fillna(0)  # Replace NaN values with 0


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


def load_results_subtle(project_dir, model_name):
    """
    Load the subclusters.csv files from the specified model directory and return the results as a dictionary.

    Parameters:
    project_dir (str): The directory containing the model directories.
    model_name (str): The name of the model directory.

    Returns:
    results_dict: A dictionary containing the subclusters data for each recording.
    """
    results_dict = {}
    
    model_dir = os.path.join(project_dir, model_name)
    
    # traverse all directories within the model directory (i.e., recording_names)
    for recording_name in os.listdir(model_dir):
        recording_path = os.path.join(model_dir, recording_name)
        
        # Verify that the current path is a directory
        if os.path.isdir(recording_path):
            subclusters_file = os.path.join(recording_path, 'subclusters.csv')
            
            # Verify that the subclusters.csv file exists
            if os.path.isfile(subclusters_file):
                subclusters_data = pd.read_csv(subclusters_file, header=None)
                subclusters_array = subclusters_data.values.flatten()   # convert to 1D array (same with kp-moseq format)
                
                results_dict[recording_name] = {'syllable': subclusters_array}
    
    return results_dict


def concatenate_stateseqs(stateseqs, mask=None):
    """
    Concatenate state sequences, optionally applying a mask.

    Parameters
    ----------
    stateseqs: ndarray of shape (..., t), or dict or list of such arrays
        Batch of state sequences where the last dim indexes time, or a
        dict/list containing state sequences as 1d arrays.

    mask: ndarray of shape (..., >=t), default=None
        Binary indicator for which elements of `stateseqs` are valid,
        used in the case where `stateseqs` is an ndarray. If `mask`
        contains more time-points than `stateseqs`, the initial extra
        time-points will be ignored.

    Returns
    -------
    stateseqs_flat: ndarray
        1d array containing all state sequences
    """
    if isinstance(stateseqs, dict):
        stateseq_flat = np.hstack(list(stateseqs.values()))
    elif isinstance(stateseqs, list):
        stateseq_flat = np.hstack(stateseqs)
    elif mask is not None:
        stateseq_flat = stateseqs[mask[:, -stateseqs.shape[1] :] > 0]
    else:
        stateseq_flat = stateseqs.flatten()
    return stateseq_flat


def get_frequencies(stateseqs, mask=None, num_states=None, runlength=True):
    """
    Get state frequencies for a batch of state sequences.

    Parameters
    ----------
    stateseqs: ndarray of shape (..., t), or dict or list of such arrays
        Batch of state sequences where the last dim indexes time, or a
        dict/list containing state sequences as 1d arrays.

    mask: ndarray of shape (..., >=t), default=None
        Binary indicator for which elements of `stateseqs` are valid,
        used in the case where `stateseqs` is an ndarray. If `mask`
        contains more time-points than `stateseqs`, the initial extra
        time-points will be ignored.

    num_states: int, default=None
        Number of different states. If None, the number of states will
        be set to `max(stateseqs)+1`.

    runlength: bool, default=True (빈도로 계산할지,  duration으로 계산할지)
        Whether to count frequency by the number of instances of each
        state (True), or by the number of frames in each state (False).

    Returns
    -------
    frequencies: 1d array
        Frequency of each state across all state sequences

    Examples
    --------
    >>> stateseqs = {
        'name1': np.array([1, 1, 2, 2, 2, 3]),
        'name2': np.array([0, 0, 0, 1])}
    >>> get_frequencies(stateseqs, runlength=True)
    array([0.2, 0.4, 0.2, 0.2])
    >>> get_frequencies(stateseqs, runlength=False)
    array([0.3, 0.3, 0.3, 0.1])
    """
    stateseq_flat = concatenate_stateseqs(stateseqs, mask=mask).astype(int)

    if runlength:
        state_onsets = np.pad(np.diff(stateseq_flat).nonzero()[0] + 1, (1, 0))
        stateseq_flat = stateseq_flat[state_onsets]

    counts = np.bincount(stateseq_flat, minlength=num_states)
    frequencies = counts / counts.sum()
    return frequencies


def compute_subtle_df(project_dir, model_name, *, fps=30, index_filename="index.csv"):
    """Compute moseq dataframe from results dict that contains all kinematic
    values by frame.

    Parameters
    ----------
    project_dir : str
        the path to the project directory
    model_name : str
        the name of the model directory
    results_dict : dict
        dictionary of results from model fitting
    use_bodyparts : bool
        boolean flag whether to include data for bodyparts

    Returns
    -------
    subtle_df : pandas.DataFrame
        the dataframe that contains kinematic data for each frame
    """

    # load model results
    results_dict = load_results_subtle(project_dir, model_name)

    # load index file
    index_filepath = os.path.join(project_dir, index_filename)
    if os.path.exists(index_filepath):
        index_data = pd.read_csv(index_filepath, index_col=False)
    else:
        print(
            "index.csv not found, if you want to include group information for each video, please run the Assign Groups widget first"
        )

    recording_name = []
    syllable = []
    frame_index = []
    s_group = []

    for k, v in results_dict.items():
        n_frame = v["syllable"].shape[0]
        recording_name.append([str(k)] * n_frame)

        if index_data is not None:
            # find the group for each recording from index data
            s_group.append(
                [index_data[index_data["name"] == k]["group"].values[0]] * n_frame
            )
        else:
            # no index data
            s_group.append(["default"] * n_frame)
        frame_index.append(np.arange(n_frame))
        
        # add syllable data
        syllable.append(v["syllable"])

    # construct dataframe
    subtle_df = pd.DataFrame(np.concatenate(recording_name), columns=["name"])
    subtle_df["syllable"] = np.concatenate(syllable)
    subtle_df["frame_index"] = np.concatenate(frame_index)
    subtle_df["group"] = np.concatenate(s_group)

    # compute syllable onset
    change = np.diff(subtle_df["syllable"]) != 0
    indices = np.where(change)[0]
    indices += 1
    indices = np.concatenate(([0], indices))

    onset = np.full(subtle_df.shape[0], False)
    onset[indices] = True
    subtle_df["onset"] = onset
    return subtle_df


def compute_stats_subtle_df(
    project_dir,
    model_name,
    subtle_df,
    min_frequency=0.001,
    groupby=["group", "name"],
    fps=30,
    index_filename="index.csv"
):
    """Summary statistics for syllable frequencies and kinematic values.

    Parameters
    ----------
    subtle_df : pandas.DataFrame
        the dataframe that contains kinematic data for each frame
    threshold : float, optional
        usge threshold for the syllable to be included, by default 0.005
    groupby : list, optional
        the list of column names to group by, by default ['group', 'name']
    fps : int, optional
        frame per second information of the recording, by default 30

    Returns
    -------
    stats_df : pandas.DataFrame
        the summary statistics dataframe for syllable frequencies and kinematic values
    """
    # compute runlength encoding for syllable

    # load model results
    results_dict = load_results_subtle(project_dir, model_name)
    syllable = {k: res["syllable"] for k, res in results_dict.items()}
    # frequencies is array of frequencies for sorted syllable [syll_0, syll_1...]
    frequencies = get_frequencies(syllable)
    syll_include = np.where(frequencies > min_frequency)[0]

    # add group information
    # load index file
    index_filepath = os.path.join(project_dir, index_filename)
    if os.path.exists(index_filepath):
        index_df = pd.read_csv(index_filepath, index_col=False)
    else:
        print(
            "index.csv not found, if you want to include group information for each video, please run the Assign Groups widget first"
        )

    # construct frequency dataframe
    # syllable frequencies within one session add up to 1
    frequency_df = []
    for k, v in results_dict.items():
        syll_freq = get_frequencies(v["syllable"])
        df = pd.DataFrame(
            {
                "name": k,
                "group": index_df[index_df["name"] == k]["group"].values[0],
                "syllable": np.arange(len(syll_freq)),
                "frequency": syll_freq,
            }
        )
        frequency_df.append(df)
    frequency_df = pd.concat(frequency_df)
    if "name" not in groupby:
        frequency_df.drop(columns=["name"], inplace=True)

    # filter out syllable that are used less than threshold in all recordings
    filtered_df = subtle_df[subtle_df["syllable"].isin(syll_include)].copy()

    # TODO: hard-coded heading for now, could add other scalars
    features = filtered_df.groupby(groupby + ["syllable"]).size().reset_index().drop(columns=0)
    
    # get durations
    trials = filtered_df["onset"].cumsum()
    trials.name = "trials"
    durations = filtered_df.groupby(groupby + ["syllable"] + [trials])["onset"].count()
    # average duration in seconds
    durations = durations.groupby(groupby + ["syllable"]).mean() / fps
    durations.name = "duration"
    # only keep the columns we need
    durations = durations.fillna(0).reset_index()[groupby + ["syllable", "duration"]]

    stats_df = pd.merge(features, frequency_df, on=groupby + ["syllable"])
    stats_df = pd.merge(stats_df, durations, on=groupby + ["syllable"])
    return stats_df