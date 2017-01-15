import pandas
import numpy as np
import matplotlib.pyplot as plt
from data_augmentation import read_pickle, read_img, flip_img, rotate_img, get_roi

rows, cols, ch = 42, 160, 3

driving_data_frame = pandas.read_csv('data/driving_log.csv')
num_rows = len(driving_data_frame)


def image_generator(pd_data_frame):
    driving_log = pd_data_frame.sample(frac=1.0)
    for i in driving_log.itertuples():
        center_cam_path = i.center.strip()
        left_cam_path = i.left.strip()
        steering_angle = i.steering
        center_cam_img = read_img(center_cam_path)
        left_cam_img = read_img(left_cam_path)
        yield center_cam_img, steering_angle
        yield left_cam_img, steering_angle + 0.25


def batch_generator(batch_size=32):
    img_dataset = np.array((batch_size, None, None, None))
    steering_dataset = np.zeros(batch_size)
    counter = -1
    while True:        
        for i in range(batch_size):
            if counter == -1 or counter >= num_rows:
                counter = 0
                img_gen = image_generator(driving_data_frame)
            img_dataset[i], steering_dataset[i] = next(img_gen)
            counter += 1
        yield img_dataset, steering_dataset


def main():
    # test
    num = 3
    for i, j in batch_generator(num):
        print(i.shape)
        print(j.shape)
        for x in range(num):
            print(i[x].shape)
            plt.subplot(1, num, x+1)
            plt.imshow(i[x], aspect='auto')
            plt.title("index: {0} steering: {1}".format(x, j[x]))
        plt.show()
        break

if __name__ == '__main__':
    main()


