from helper import getAllWithRawFolders

def get_train_sets():
    return getAllWithRawFolders(r'datasets/train')

def get_val_sets():
    return getAllWithRawFolders(r'datasets/val')

def get_test_sets():
    return getAllWithRawFolders(r'datasets/test')
