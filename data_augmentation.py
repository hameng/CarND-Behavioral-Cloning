from PIL import Image
import scipy.ndimage
import pandas
import numpy as np
import matplotlib.image as mpimg
import pickle
import matplotlib.pyplot as plt
import random
import cv2

center_cam_data_file = 'data/center_cam_data.p'
rotated_data_file = 'data/rotated_data.p'
flipped_data_file = 'data/flipped_data.p'
left_cam_data_file = 'data/left_cam_data.p'
right_cam_data_file = 'data/right_cam_data.p'

# crop and resize the image from 160x320 to 42x160
def get_roi(img_data):
    cropped_image = img_data[55:-20, :, :] 
    image = cv2.resize(cropped_image, (0,0), fx=0.5, fy=0.5)
    return image

# thanks to Vivek Yadav
# change the brightness with a random number
def change_brightness(img_data):
    # convert to HSV so that its easy to adjust brightness
    image1 = cv2.cvtColor(img_data, cv2.COLOR_RGB2HSV)
    
    # randomly generate the brightness reduction factor
    # Add a constant so that it prevents the image from being completely dark
    random_bright = 0.25 + np.random.uniform()

    # Apply the brightness reduction to the V channel
    image1[:, :, 2] = image1[:, :, 2] * random_bright
    # convert to RBG again
    return cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)


# read the image from a file into an array
def read_img(img_path):
    img_data = mpimg.imread(img_path)
    return get_roi(img_data)


# flip the image in the left/right direction
def flip_img(img_data):
    return np.fliplr(img_data)


# roate image in a random angle [-8, -6, 6, 8]
def rotate_img(img_data):
    rotate_angle = random.sample([-8, -6, 6, 8], 1)[0]
    return scipy.ndimage.rotate(img_data, rotate_angle, reshape=False, mode='nearest')



def augment_data(csv_path='data/driving_log.csv'):
    csv_data = pandas.read_csv(
        csv_path, header=0,
        names=['center', 'left', 'right', 'steering', 'throttle', 'break', 'speed']
    )
    print(csv_data['center'].values[0])
    # data from center_camera without brightness changing
    x_train_center_cam = np.array([read_img(img_path.strip()) for img_path in csv_data['center'].values])
    # data from left camera with random brightness changing
    x_train_left_cam = np.array([change_brightness(read_img(img_path.strip())) for img_path in csv_data['left'].values])
    # data from right camera with random brightness changing
    x_train_right_cam = np.array([change_brightness(read_img(img_path.strip())) for img_path in csv_data['right'].values])
    # rotate the center camera images then change the brightness randomly
    x_train_rotate_center = np.array([change_brightness(rotate_img(img_data)) for img_data in x_train_center_cam])
    # flip the center camera images then change the brightness randomly
    x_train_flip_center = np.array([change_brightness(flip_img(img_data)) for img_data in x_train_center_cam])
    # change the brightness randomly in center camera dataset
    x_train_center_cam = np.array([change_brightness(img_data) for img_data in x_train_center_cam])

    y_train_origin = csv_data['steering'].values

    assert x_train_center_cam.shape == x_train_left_cam.shape == x_train_right_cam.shape == \
           x_train_flip_center.shape == x_train_rotate_center.shape
    assert y_train_origin.shape[0] == x_train_flip_center.shape[0]

    print(x_train_flip_center.shape, y_train_origin.shape)

    with open(center_cam_data_file, mode='wb') as w:
        pickle.dump({"img": x_train_center_cam, "steering": y_train_origin}, w)
    
    with open(rotated_data_file, mode='wb') as w:
        pickle.dump({"img": x_train_rotate_center, "steering": y_train_origin}, w)
    
    with open(flipped_data_file, mode='wb') as w:
        pickle.dump({"img": x_train_flip_center, "steering": -y_train_origin}, w)

    # the steering angle from left camera should be "turn right" a little bit
    with open(left_cam_data_file, mode='wb') as w:
        pickle.dump({"img": x_train_left_cam, "steering": y_train_origin + 0.250}, w)
    # the steering angle from right camera should be "turn left" a little bit
    with open(right_cam_data_file, mode='wb') as w:
        pickle.dump({"img": x_train_right_cam, "steering": y_train_origin - 0.250}, w)
 

def read_pickle(pickle_file_path):
    with open(pickle_file_path, mode='rb') as f:
        data_dict = pickle.load(f)
        return data_dict['img'], data_dict['steering']


def show_stat_plot(steering_data, title):
    plt.figure(figsize=(5,2))
    plt.hist(steering_data*10,  bins=80, alpha=0.5)
    plt.title(title)
    plt.xlabel('Steering angle x 10')
    plt.ylabel('count')
    plt.show()


def verify_data_process():
    center_cam_data, origin_steering = read_pickle(center_cam_data_file)
    left_cam_data, left_cam_steering = read_pickle(left_cam_data_file)
    right_cam_data, right_cam_steering = read_pickle(right_cam_data_file)
    rotate_img_data, rotate_steering = read_pickle(rotated_data_file)
    flip_img_data, flip_steering = read_pickle(flipped_data_file)

    i = 0
    while i<15:
        index = random.randrange(5000) 
        for img_data, steering_data in [(center_cam_data[index], origin_steering[index]),
        (rotate_img_data[index], rotate_steering[index]),
        (flip_img_data[index], flip_steering[index]),
        (left_cam_data[index], left_cam_steering[index]),
        (right_cam_data[index], right_cam_steering[index])]:
            plt.subplot(3, 5, i+1)
            plt.imshow(img_data, aspect='auto')
            plt.title("{0} steering:{1}".format(index, steering_data))
            i += 1
    plt.show()

    show_stat_plot(origin_steering, "center camera")
    show_stat_plot(left_cam_steering, "left camera")
    show_stat_plot(right_cam_steering, "right camera")
    show_stat_plot(flip_steering, "flip")


def main():
    #augment_data("data/driving_log.csv")
    verify_data_process()


if __name__ == '__main__':
    main()


