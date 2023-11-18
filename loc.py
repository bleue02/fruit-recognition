import os
def root():
    train_loc = '/Users/merbleue/Desktop/fruit/fruits-360_dataset/fruits-360/Training'
    test_loc = '/Users/merbleue/Desktop/fruit/fruits-360_dataset/fruits-360/Test'
    return train_loc, test_loc
root()



# def split_Data():
#     train_dir = os.path.join(os.getcwd(), 'train')
#     test_dir = os.path.join(os.getcwd(), 'test')
#
#     train_set_dir = os.path.join(train_dir, 'train')
#     os.mkdir(train_set_dir)
#     valid_set_dir = os.path.join(train_dir, 'valid')
#     os.mkdir(valid_set_dir)
#     test_set_dir = os.path.join(train_dir, 'test')
#     os.mkdir(test_set_dir)
#
#
