## General
project_name = 'SDSBD'
dataset_name = r'C:\Users\MyPC\Desktop\실험실\2.실험데이터\AVATAR-SDSBD\dataset4'
fs = 20  # sampling rate of the video data
columns_to_remove = [24, 25, 26]  # columns to be removed from the data (tail tip in AVATAR data)

## subtle
TRAIN_DIR = r"C:\Users\MyPC\Desktop\git\SUBTLE_June\project\SDSBD\trainset1.txt"
TRAIN_SAVE_DIR = r"C:\Users\MyPC\Desktop\git\SUBTLE_June\project\SDSBD\model2\results_trainset1"

TEST_DIR = r"C:\Users\MyPC\Desktop\git\SUBTLE_June\project\SDSBD\testset1.txt"
TEST_SAVE_DIR = r"C:\Users\MyPC\Desktop\git\SUBTLE_June\project\SDSBD\model2\results_testset1"

## after_subtle
min_occurrence = 0.01  # minimum occurrence of a subcluster to be considered as a cluster
subcluster_num = 74

DIR_preSD_C = r'C:\Users\MyPC\Desktop\git\after_subtle\Project\SDSBD\results1\preSD_C.txt'
DIR_preSD_R = r'C:\Users\MyPC\Desktop\git\after_subtle\Project\SDSBD\results1\preSD_R.txt'
DIR_preSD_S = r'C:\Users\MyPC\Desktop\git\after_subtle\Project\SDSBD\results1\preSD_S.txt'
DIR_preSD_S_extreme = r'C:\Users\MyPC\Desktop\git\after_subtle\Project\SDSBD\results1\preSD_S_extreme.txt'

SAVE_DIR_pre = r'C:\Users\MyPC\Desktop\git\after_subtle\Project\SDSBD\results1\group_comparison\preSD'
SAVE_DIR_post = r'C:\Users\MyPC\Desktop\git\after_subtle\Project\SDSBD\results1\group_comparison\preSD'

COORDS_DIR_pre = r'C:\Users\MyPC\Desktop\실험실\2.실험데이터\AVATAR-SDSBD\dataset4\preSD'
COORDS_DIR_post = r'C:\Users\MyPC\Desktop\실험실\2.실험데이터\AVATAR-SDSBD\dataset4\postSD'