import os
from PIL import Image
import numpy as np

root = './../DATA/'

class_dir = os.path.join(root,"CUB_200_2011","images")
class_list = os.listdir(class_dir)
for cla in class_dir:#遍历每一个文件夹
    image_list = os.listdir(os.path.join(class_dir,cla))
    index = np.random.randint(0,len(image_list)-1,size=(3))
    for i in index:#遍历每个文件夹下随机的3个图片
        img_path = os.path.join(class_dir,cla,image_list[i])
        class_number = cla.split('.')[0]
        if not os.path.exists(os.path.join(root,'test')):
            os.mkdir(os.path.join(root,'test'))

        img_save_path = os.path.join(root,'test')
        img_save_path = img_save_path+"/"+cla+image_list[i]
        img = Image.open(img_path)
        #img.save(img_save_path)
        print(img_save_path)



