import json
import sys
import numpy as np
import random
#import pickle
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, ELU, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.utils import shuffle
#from sklearn.model_selection import train_test_split
from data_augmentation import read_pickle, show_stat_plot
import matplotlib.pyplot as plt
import tensorflow as tf
tf.python.control_flow_ops = tf

SHOW_PLOT = False
rows, cols, ch = 42, 160, 3

center_cam_data_file = 'data/center_cam_data.p'
rotated_data_file = 'data/rotated_data.p'
flipped_data_file = 'data/flipped_data.p'
left_cam_data_file = 'data/left_cam_data.p'
right_cam_data_file = 'data/right_cam_data.p'


def find_largest_count(np_data_set):
    np_round = np.round(np_data_set*10, 2)
    unique, counts = np.unique(np_round, return_counts=True)
    np_count = np.asarray((unique, counts)).T
    largest = np_count[np_count[:,1].argsort()]
    

    zero_count = np.where(np_data_set==0)[0].shape[0]
    negative_count = np.where(np_data_set<0.00)[0].shape[0]
    positive_count = np.where(np_data_set>0.00)[0].shape[0]
    print(largest[-6:])
    print("0 count: {0}, - count {1}, + count {2}".format(zero_count, negative_count, positive_count))


def remove_data(steering_data, img_data, val=0.0, drop=0.75, is_round=False):
    if is_round:
        target_value_index = np.where(np.round(steering_data, 2)==val)[0]
    else:
        target_value_index = np.where(steering_data==val)[0]
    to_be_delete = random.sample(list(target_value_index), int(len(target_value_index)*drop))
    steering_data = np.delete(steering_data, to_be_delete)
    img_data = np.delete(img_data, to_be_delete, axis=0)
    return steering_data, img_data


def reprocess_data():
    center_cam_data, origin_steering = read_pickle(center_cam_data_file)
    rotate_img_data, rotate_steering = read_pickle(rotated_data_file)
    flip_img_data, flip_steering = read_pickle(flipped_data_file)
    left_cam_data, left_cam_steering = read_pickle(left_cam_data_file)
    right_cam_data, right_cam_steering = read_pickle(right_cam_data_file)
    
    # concatenate all image data
    img_data = np.concatenate((center_cam_data, rotate_img_data, flip_img_data, left_cam_data))#, right_cam_data))
    # concatenate all steering data
    steering_data = np.concatenate((origin_steering, rotate_steering, flip_steering, left_cam_steering))#, right_cam_steering))
    
    print(img_data.shape, steering_data.shape)
    find_largest_count(steering_data)
    if SHOW_PLOT:
        show_stat_plot(steering_data, "concatenate")
        

    steering_data, img_data = remove_data(steering_data, img_data, val=0, drop=0.95)
    print("after removing 95% 0 angle steering data ...", img_data.shape, steering_data.shape)
    find_largest_count(steering_data)
    if SHOW_PLOT:
        show_stat_plot(steering_data, "after removing 95% 0 angle steering data")
    
    steering_data, img_data = remove_data(steering_data, img_data, val=0.25, drop=0.88, is_round=True)
    print("after removing 88% 0.25 steerings data ...",img_data.shape, steering_data.shape)
    find_largest_count(steering_data)
    if SHOW_PLOT:
        show_stat_plot(steering_data, "after removing 88% +0.25 angle steering data")

    #steering_data, img_data = remove_data(steering_data, img_data, val=-0.25, drop=0.80, is_round=True)
    #print("after removing 75% 0.25 steerings data ...",img_data.shape, steering_data.shape)
    #print(find_largest_count(steering_data*10))
    #if True:
        #show_stat_plot(steering_data, "after removing 75% -0.25 angle steering data")
     
    # shuffle the dataset
    train_x, train_y = shuffle(img_data, steering_data) 
    if SHOW_PLOT:
        i = 0
        while i<9:
        #for i in range(15):
            index = random.randrange(train_y.shape[0]) 
            for img_data, steering_data in [(train_x[index], train_y[index])]:
                plt.subplot(3, 3, i+1)
                plt.imshow(img_data, aspect='auto')
                plt.title(steering_data)
                i += 1
        plt.show()
    return train_x,  train_y

    
def get_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.50,
                input_shape=(rows, cols, ch),
                output_shape=(rows, cols, ch)))

    model.add(Convolution2D(3, 1, 1, input_shape=(rows, cols, ch)))

    model.add(Convolution2D(16, 5, 5, input_shape=(rows, cols, ch)))
    model.add(ELU())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Convolution2D(32, 5, 5)) #, subsample=(2, 2)))
    model.add(ELU())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Convolution2D(48, 3, 3)) #, subsample=(2, 2)))
    model.add(ELU())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(ELU())

    model.add(Dense(128))
    model.add(Dropout(0.3))
    model.add(ELU())

    model.add(Dense(64))
    model.add(Dropout(0.3))
    model.add(ELU())

    model.add(Dense(16))
    model.add(Dropout(0.3))
    model.add(ELU())

    model.add(Dense(1))
    return model


def main():
    model = get_model()
    model.compile(loss='mean_squared_error', optimizer='adam')#, metrics=['accuracy'])
    print(model.summary())
    train_features, train_steerings  = reprocess_data()
    model.fit(train_features, train_steerings, batch_size=256, nb_epoch=10, validation_split=0.20)

    with open('model.json', 'w') as fd:
        json.dump(model.to_json(), fd)
    model.save_weights('model.h5')

if __name__ == '__main__':
    main()
