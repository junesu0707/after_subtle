## General
project_name = 'BDB'
dataset_name = r'C:\Users\MyPC\Desktop\실험실\2.실험데이터\AVATAR-BDB\mirror_game_concat_processed'
fs = 30  # sampling rate of the video data
columns_to_remove = []  # columns to be removed from the data (tail tip in AVATA  R data)

## subtle
# TRAIN_DIR = r"C:\Users\MyPC\Desktop\git\SUBTLE_June\project\SDSBD\trainset1.txt"
# TRAIN_SAVE_DIR = r"C:\Users\MyPC\Desktop\git\SUBTLE_June\project\SDSBD\model2\results_trainset1"

# TEST_DIR = r"C:\Users\MyPC\Desktop\git\SUBTLE_June\project\SDSBD\testset1.txt"
# TEST_SAVE_DIR = r"C:\Users\MyPC\Desktop\git\SUBTLE_June\project\SDSBD\model2\results_testset1"

## after_subtle
min_occurrence = 0.01  # minimum occurrence of a subcluster to be considered as a cluster
subcluster_num = 85
supercluster_num = 4

DIR_like_y = r'C:\Users\MyPC\Desktop\git\after_subtle\Project\BDB\results1\like_y.txt'
DIR_like_n = r'C:\Users\MyPC\Desktop\git\after_subtle\Project\BDB\results1\like_n.txt'

SAVE_DIR_like = r'C:\Users\MyPC\Desktop\git\after_subtle\Project\BDB\results1\group_comparison\like'

COORDS_DIR_like = r'C:\Users\MyPC\Desktop\실험실\2.실험데이터\AVATAR-BDB\mirror_game_concat_processed'