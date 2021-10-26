import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

amount_of_images=50000
target_resolution=128


image_folder_path='C:/Users/Dominic/Downloads/img_align_celeba/img_align_celeba/'

file_names=os.listdir(image_folder_path)
file_paths=[os.path.join(image_folder_path,f) for f in file_names]
images=[]
i=0
for file_path in file_paths:
    i+=1
    if i%100==0:
        print(i)

    if len(images) >= amount_of_images:
        break
    img=cv2.imread(file_path)
    img = img[20:-20, :, :]
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(target_resolution,target_resolution))
    images.append(img)
    # plt.figure()
    # plt.imshow(img)
    # plt.show()


images=np.array(images)
np.save('./data/celeba_128.npy',images)
