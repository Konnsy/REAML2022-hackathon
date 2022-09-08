import os
import wget
from pyunpack import Archive

def download_archive():
    wget.download("https://tu-dortmund.sciebo.de/s/Tj7665zMS62Gfkz/download") #Change url to wherever the data is
    os.rename('download','download.7z')
    os.mkdir('rawdata')
    Archive('download.7z').extractall("rawdata")
    Archive("rawdata/datasets_task1_600_per_set/dataset_pos.7z").extractall(os.getcwd())
    Archive("rawdata/datasets_task1_600_per_set/dataset_neg.7z").extractall(os.getcwd())
    Archive("rawdata/datasets_task1_600_per_set/dataset_pos_val.7z").extractall(os.getcwd())
    Archive("rawdata/datasets_task1_600_per_set/dataset_neg_val.7z").extractall(os.getcwd())


if __name__ == '__main__':
    download_archive()

