import os

from PIL import Image, ImageOps
import numpy as np
import scipy.misc
from scipy.misc import imread
import cv2
import pdb

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# data_dir = "/home/abhishek/Downloads/CIS680_2017Fall-master/hw2.b/P&C dataset"

def parse_data(path, faltten):
    #load training images
    # pdb.set_trace()
    print('Loading data ')
    image_path = os.path.join(path, "img")
    num_images = 2000
    X = np.zeros((num_images, 128, 128, 3), dtype=np.float32)

    with os.scandir(image_path) as it:
        i = 0
        for entry in it:
            if not entry.name.startswith('.') and entry.is_file():
                img_file = os.path.join(image_path, entry)
                img = imread(img_file)
                names = img_file.split('/')
                file_names = names[-1].split('.')
                file_num = int(file_names[0])
                X[file_num] = img
                i += 1

                # X[0] = img.transpose(2, 0, 1)
        # np.save(os.path.join(path, "X.npy"), X)
        # print("Total images loaded: ", i)

    #load masks
    people_mask_path = os.path.join(path, "mask", "people")
    car_mask_path = os.path.join(path, "mask", "car")
    car_mask = []
  
    for j in range(num_images):
        img_file = os.path.join(car_mask_path, "{0:06d}.png".format(j))
        img = imread(img_file)
        car_mask.append(img)
    car_mask = np.array(car_mask)
    # np.save(os.path.join(path,"car_mask.npy"), car_mask)

    people_mask = []
    for j in range(num_images):
        img_file = os.path.join(people_mask_path, "{0:06d}.png".format(j))
        img = imread(img_file)
        people_mask.append(img)
    people_mask = np.array(people_mask)
    # np.save(os.path.join(path,"people_mask.npy"), people_mask)

    #load labels
    label_people_file = os.path.join(path, "label_people.txt")
    people_label_file_handle = open(label_people_file, 'r')
    people_label = np.zeros((num_images, 4))
    i = 0
    for line in people_label_file_handle:
        strs = line.split(',')
        for j in range(len(strs)):
            people_label[i, j] = int(strs[j])
        i += 1
    people_label_file_handle.close()
    # np.save(os.path.join(path,"label_people.npy"), people_label)

    label_car_file = os.path.join(path, "label_car.txt")
    car_label_file_handle = open(label_car_file, 'r')
    car_label = np.zeros((num_images, 4))
    i = 0
    for line in car_label_file_handle:
        strs = line.split(',')
        for j in range(len(strs)):
            car_label[i, j] = int(strs[j])
        i += 1
    car_label_file_handle.close()
    # np.save(os.path.join(path,"label_car.npy"), car_label)
    print("Done loading data")

    return X, people_mask, car_mask, people_label, car_label

               
# parse_data(data_dir, False)

def read_data(path, flatten=True, num_train=1600):
    """
    Read in the dataset, given that the data is stored in path
    Return two tuples of numpy arrays
    ((train_imgs, train_people_mask,... ), (val_imgs, val_people_mask, ...))
    """
   
    imgs, people_mask, car_mask, people_label, car_label = parse_data(path, flatten)
    np.random.seed(42)
    indices = np.random.permutation(imgs.shape[0])
    train_idx, val_idx = indices[:num_train], indices[num_train:]
    
    train_img, train_people_mask, train_car_mask, train_people_label, train_car_label = imgs[train_idx, :], people_mask[train_idx], car_mask[train_idx], people_label[train_idx, :], car_label[train_idx, :]
   
    val_img, val_people_mask, val_car_mask, val_people_label, val_car_label = imgs[val_idx, :], people_mask[val_idx], car_mask[val_idx], people_label[val_idx, :], car_label[val_idx, :]
    
    return (train_img, train_people_mask, train_car_mask, train_people_label, train_car_label), (val_img, val_people_mask, val_car_mask, val_people_label, val_car_label)

class DataIter(object):
    def __init__(self, dataTuple, anchor):

        self.img, self.people_mask, self.car_mask, self.people_label, self.car_label = dataTuple
       
        self.people_mask, self.car_mask = self.reshapeMask(self.people_mask), self.reshapeMask(self.car_mask)
       
        self.anchor = anchor
    
    def genData(self):
        for i in range(self.img.shape[0]):
           
            # pdb.set_trace()
            iou_scores, bbox_matrix, tx_star, ty_star, tw_star, th_star, label = construct_iou_matrix(self.car_label[i,:]/16, self.people_label[i,:]/16, self.anchor)
            thres = .5
            iou_scores = np.where(iou_scores>thres, 1, iou_scores)
            iou_scores = np.where(iou_scores<.1, 0, iou_scores)
            iou_scores = np.where((iou_scores <= thres) * ( iou_scores >= .1) , -1, iou_scores)
            
            yield self.img[i,:],  self.people_label[i,:], self.car_label[i,:], iou_scores, bbox_matrix, tx_star, ty_star, tw_star, th_star, label, self.people_mask[i,:], self.car_mask[i,:]

    def reshapeMask(self, masks):
        mask_reshaped = np.zeros(shape=[masks.shape[0], 22, 22])
        for i in range(masks.shape[0]):
            mask = cv2.resize(masks[i], (22,22))
            # mask = mask.reshape(22,22,1)
            mask_reshaped[i] = mask

        return mask_reshaped.reshape(-1,22,22,1)

    

def get_dataset(batch_size, anchor):
    # Step 1: Read in data
    data_folder = "/home/abhishek/Downloads/CIS680_2017Fall-master/hw2.b/P&C dataset"
    
    train, val= read_data(data_folder, flatten=False)
    # pdb.set_trace()
    # Step 2: Create datasets and iterator
    train = DataIter(train,anchor=anchor)
    val = DataIter(val,anchor = anchor)
    train_data = tf.data.Dataset.from_generator(train.genData,
                                                (tf.float32,  tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.int32, tf.float32, tf.float32),
                                                ((128,128,3), (4,), (4,), (8,8,1), (8,8,4), (8,8,1), (8,8,1), (8,8,1), (8,8,1), (8,8,1), (22,22,1), (22,22,1) ))
    train_data = train_data.batch(batch_size)
    
    val_data = tf.data.Dataset.from_generator(val.genData,
                                                (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.int32, tf.float32, tf.float32),
                                                ((128,128,3), (4,), (4,), (8,8,1), (8,8,4), (8,8,1), (8,8,1), (8,8,1), (8,8,1), (8,8,1), (22,22,1), (22,22,1) ))
   
    val_data = val_data.batch(batch_size)
    # pdb.set_trace()
    return train_data, val_data


def iou(box_A, box_B):
    x11 = box_A[0]
    y11 = box_A[1]
    w1 = box_A[2]
    h1 = box_A[3]
    # x12, y12 = x11, y11 + w1
    x13, y13 = x11 + w1, y11 + h1
    # x14, y14 = x11 + h1, y11

    x21 = box_B[0]
    y21 = box_B[1]
    w2 = box_B[2]
    h2 = box_B[3]
    # x22, y22 = x21, y21 + w2
    x23, y23 = x21 + w2, y21 + h2
    # x24, y24 = x21 + h2, y21

    xA = np.maximum(x11, x21)
    yA = np.maximum(y11, y21)
    xB = np.minimum(x13, x23)
    yB = np.minimum(y13, y23)

    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    # boxAArea for anchor
    boxAArea = (x13 - x11 +1) * (y13 - y11 +1) 
    boxBArea = (x23 - x21 ) * (y23 - y21 )

    iou = interArea / (boxAArea + boxBArea - interArea) 

    return iou

def construct_iou_matrix(car_label, people_label, anchor):
    # pdb.set_trace()
    '''
    bbox_matrix x,y coordinates are for the center of  the box
    bbox_matrix has x,y,w,h values for each location of the anchor
    label : 0 for people and 1 for car
    '''
    w_a, h_a = anchor[2] + 1, anchor[3] + 1
    people = 0
    car = 1
    tx_star = np.zeros((8,8))
    ty_star = np.zeros((8,8))
    tw_star = np.zeros((8,8))
    th_star = np.zeros((8,8))
    A = np.zeros((8,8))
    label = np.zeros((8,8))
    bbox_matrix = np.zeros((8,8,4))
    # pdb.set_trace()
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            # pdb.set_trace()
            iou_car = iou(anchor + np.array([i, j, 0, 0 ]), car_label)
            iou_people = iou(anchor + np.array([i, j, 0, 0 ]), people_label)
            if (iou_car > iou_people):
                A[j,i] = iou_car
                label[j,i] = car
                tx_star[j,i] = (car_label[0] - anchor[0] - i - 1)/w_a
                ty_star[j,i] = (car_label[1] - anchor[1] - j - 1)/w_a
                tw_star[j,i] = np.log(1e-9 + car_label[2] / w_a)
                th_star[j,i] = np.log(1e-9 + car_label[3] / h_a)
            else :
                A[j,i] = iou_people
                label[j,i] = people
                tx_star[j,i] = (people_label[0] - anchor[0] - i - 1)/w_a
                ty_star[j,i] = (people_label[1] - anchor[1] - j - 1)/w_a
                tw_star[j,i] = np.log(1e-9 + people_label[2] / w_a)
                th_star[j,i] = np.log(1e-9 + people_label[3] / h_a)
           
            bbox_matrix[j,i,0] = anchor[0] + i + 1
            bbox_matrix[j,i,1] = anchor[1] + j + 1
            bbox_matrix[j,i,2] = w_a
            bbox_matrix[j,i,3] = h_a
    # pdb.set_trace()
    return A.reshape(8,8,1), bbox_matrix, tx_star.reshape((8,8,1)), ty_star.reshape((8,8,1)), tw_star.reshape((8,8,1)), th_star.reshape((8,8,1)), label.reshape(8,8,1)