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
subcluster_num = 74

DIR_gria3 = r'C:\Users\MyPC\Desktop\git\SUBTLE_June\project\Broad\model1\results_testset1\Gria3\group_gria3.txt'
DIR_setd1a = r'C:\Users\MyPC\Desktop\git\SUBTLE_June\project\Broad\model1\results_testset1\Setd1a\group_setd1a.txt'
DIR_gria3_wt = r'C:\Users\MyPC\Desktop\git\SUBTLE_June\project\Broad\model1\results_testset1\Gria3\group_gria3_wt.txt'
DIR_gria3_mut = r'C:\Users\MyPC\Desktop\git\SUBTLE_June\project\Broad\model1\results_testset1\Gria3\group_gria3_mut.txt'
DIR_setd1a_wt = r'C:\Users\MyPC\Desktop\git\SUBTLE_June\project\Broad\model1\results_testset1\Setd1a\group_setd1a_wt.txt'
DIR_setd1a_het = r'C:\Users\MyPC\Desktop\git\SUBTLE_June\project\Broad\model1\results_testset1\Setd1a\group_setd1a_het.txt'

SAVE_DIR_gria3 = r'C:\Users\MyPC\Desktop\git\SUBTLE_June\project\Broad\model1\results_testset1\Gria3'
SAVE_DIR_setd1a = r'C:\Users\MyPC\Desktop\git\SUBTLE_June\project\Broad\model1\results_testset1\Setd1a'

COORDS_DIR_gria3 = r'C:\Users\MyPC\Desktop\실험실\2.실험데이터\AVATAR-Broad\GRIA3-8_7\dataset1'
COORDS_DIR_setd1a = r'C:\Users\MyPC\Desktop\실험실\2.실험데이터\AVATAR-Broad\SETD1A-8_7\dataset1'