import os
from PIL import Image
import numpy as np
import imageio


def select_from_data_Set(root='./../DATA/'):
    """
    random select 3 images from each catalog in images set
    Args:
        root:
    Returns: save the selected image in DATA/Test_image/

    """
    class_dir = os.path.join(root, "CUB_200_2011", "images")
    class_list = os.listdir(class_dir)

    for cla in class_list:  # 遍历每一个文件夹
        image_list = os.listdir(os.path.join(class_dir, cla))
        index = np.random.randint(0, len(image_list) - 1, size=(3))
        for i in index:  # 遍历每个文件夹下随机的3个图片
            img_path = os.path.join(class_dir, cla, image_list[i])

            class_number = cla.split('.')[0]
            if not os.path.exists(os.path.join(root, 'Test_image')):
                os.mkdir(os.path.join(root, 'Test_image'))

            img_save_path = os.path.join(root, 'Test_image', class_number)
            img_save_path = img_save_path + image_list[i]
            print(img_save_path)
            img = Image.open(img_path)
            img.save(img_save_path)


def test_set_generate(root, is_train=False, data_len=None):
    """
    select 200 images from test set and save in Test_image
    Args:
        root:
        is_train:
        data_len:

    Returns:

    """
    img_txt_file = open(os.path.join(root, "images.txt"))
    label_txt_file = open(os.path.join(root, "image_class_labels.txt"))
    train_val_file = open(os.path.join(root, "train_test_split.txt"))
    img_name_list = []  # all the image name
    for line in img_txt_file:
        img_name_list.append(line[:-1].split(' ')[-1])
    label_list = []  # all the image label
    for line in label_txt_file:
        label_list.append(int(line[:-1].split(' ')[-1]) - 1)
    train_test_list = []  # train/test list
    for line in train_val_file:
        train_test_list.append(int(line[:-1].split(' ')[-1]))
    train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
    test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
    index_list = np.random.randint(0, len(test_file_list) - 1, size=(200))
    for index in index_list:
        image_path = os.path.join(root, "images", test_file_list[index])
        img = Image.open(image_path)
        inlist = test_file_list[index].split("/")
        inlist2 = inlist[0].split(".")
        image_name = inlist2[0] + "_" + inlist[1]
        img_save_path = os.path.join(root, "../Test_image", image_name)
        img.save(img_save_path)


test_set_generate('./../DATA/CUB_200_2011', False, None)

# select_from_data_Set(root = './../DATA/')
