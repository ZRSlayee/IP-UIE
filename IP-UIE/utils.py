import numpy as np
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

plt.switch_backend("agg")

def data_augmentation(image, mode): # 数据增强
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)

def load_images(file):
    im = Image.open(file)
    ## normalization
    img = np.array(im, dtype="float32") / 255.0
    #img_max = np.max(img)
    #img_min = np.min(img)
    #img_norm = np.float32((img - img_min)/np.maximum((img_max-img_min), 0.001))
    return img

def save_images(filepath, result_1, result_2 = None):
    result_1 = np.squeeze(result_1) # 删除数组中的单维度条目
    result_2 = np.squeeze(result_2) 

    if not result_2.any():
        cat_image = result_1
        cat_image = (cat_image - np.min(cat_image)) / (np.max(cat_image) - np.min(cat_image))### 输出前再做一次归一化
    else:
        result_1 = (result_1 - np.min(result_1)) / (np.max(result_1)-np.min(result_1))
        result_2 = (result_2 - np.min(result_2)) / (np.max(result_2)-np.min(result_2))
        cat_image = np.concatenate([result_1, result_2], axis = 1)
#     cat_image = (cat_image - np.min(cat_image)) / (np.max(cat_image) - np.min(cat_image))### 输出前再做一次归一化
    im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8')) # 将数组中的数限制到范围0-255之间
    im.save(filepath, 'png')
    
def save_heatmap(filepath, result_3):
    result_3 = np.squeeze(result_3)
    print(result_3.shape)
    min_ = np.min(np.min(result_3))
    max_ = np.max(np.max(result_3))
    hx = sns.heatmap(result_3, vmin=min_, vmax=max_, cbar=False, cmap=plt.get_cmap("jet"))
    plt.axis("off")
    plt.savefig(filepath)
    plt.close()

# def save_images(filepath, result_1, result_2 = None, result_3 = None):
#     result_1 = np.squeeze(result_1) # 删除数组中的单维度条目
#     result_2 = np.squeeze(result_2)
#     result_3 = np.squeeze(result_3)

#     if not result_3.any():
#         if not result_2.any():
#             cat_image = result_1
#         else:
#             cat_image = np.concatenate([result_1, result_2], axis = 1)
#     else:
#         cat_image = np.concatenate([result_1, result_2, result_3], axis = 1)

#     im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8')) # 将数组中的数限制到范围0-255之间
#     im.save(filepath, 'png')
