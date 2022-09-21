from helper import getAllWithRawFolders

def get_pos_train_sets():
    return getAllWithRawFolders(r'datasets/pos_train')

def get_neg_train_sets():
    return getAllWithRawFolders(r'datasets/neg_train')

def get_pos_val_sets():
    return getAllWithRawFolders(r'datasets/pos_val')

def get_neg_val_sets():
    return getAllWithRawFolders(r'datasets/neg_val')

def get_test_sets():
    return getAllWithRawFolders(r'datasets/test')