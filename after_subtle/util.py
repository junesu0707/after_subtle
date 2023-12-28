import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.mixture import GaussianMixture


def get_folder_list(directory_path):
    """Get a list of folders in the specified directory."""
    folder_list = []
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isdir(item_path):
            folder_list.append(item_path)
    return folder_list


def import_SIT_params(text_file_path):
    df = pd.read_excel(text_file_path)

    # Create a dictionary with column 1 as key and column 2 as value
    my_dict = dict(zip(df['mouse_name'], df['SIratio-postSD']))

    return my_dict


def get_complementary_color(color1):
    """
    Calculates and returns the complementary color of the given color.

    Parameters:
    color1 (tuple): The base color in RGB format.

    Returns:
    tuple: The complementary color in RGB format.
    """
    # Convert the base color to HSV
    color1_rgb = np.array([color1]).reshape(1, 3)
    color1_hsv = mcolors.rgb_to_hsv(color1_rgb)

    # Calculate the complementary color
    color2_hsv = color1_hsv.copy()
    color2_hsv[0, 0] = (color2_hsv[0, 0] + 0.5) % 1.0
    color2_rgb = mcolors.hsv_to_rgb(color2_hsv)

    return color2_rgb.flatten()
