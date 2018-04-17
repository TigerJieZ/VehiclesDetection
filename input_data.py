# -*-coding:utf-8-*-

import os
import numpy
import tensorflow as tf
import cv2
import glob


class mydata:
    def get_datas_dirs_lists(self):
        # 存放图片集的路径
        imgs = []

        # 存放男女标签，男：0，女：1
        labels = []

        # 遍历male_dir中的所有文件，来填充male_imgs和male_labels
        # for i in index:
        #     temp = []
        #     print(datas[i[0]])
        #     for path in os.listdir(datas[i[0]]):
        #         name = path.split('.')
        #         if name[-1] == 'jpg':
        #             temp.append(datas[i[0]] + path)
        #             labels.append(datas[i[1]])
        #     imgs = numpy.hstack((imgs, temp))
        # 遍历male_dir中的所有文件，来填充male_imgs和male_labels
        file_vehicles = glob.glob('E:\\vehicles\\vehicles\\*\\*\\*.png')
        label_vehicles = [1 for i in range(len(file_vehicles))]
        file_nonVehicles = glob.glob('E:\\vehicles\\non-vehicles\\*\\*\\*.png')
        label_nonVehicles = [0 for i in range(len(file_nonVehicles))]
        file_nonVehicles.extend(file_vehicles)
        label_nonVehicles.extend(label_vehicles)

        # 打乱列表元素的顺序
        def shuffle(imgs, labels):
            # 先将图片集和标签集存放在二维数组中，这样图片和对应的标签作为一个整体进行打乱
            temp = numpy.array([imgs, labels])
            # 矩阵转置
            temp = temp.transpose()
            # 打乱
            numpy.random.shuffle(temp)
            # 将temp再次分开
            imgs = temp[:, 0]
            labels = temp[:, 1]
            labels = [int(i) for i in labels]

            return imgs, labels

        imgs, labels = shuffle(imgs=file_nonVehicles, labels=label_nonVehicles)
        return imgs, labels

    def get_batch(slef, batch_size, imgs, labels, img_width, img_height, capacity):
        # 将图片路径和标签转化成ｔｅｎｓｏｒｆｌｏｗ中的数据形式
        imgs = tf.cast(imgs, tf.string)
        labels = tf.cast(labels, tf.int32)

        # 生成输入队列
        input_queue = tf.train.slice_input_producer([imgs, labels])

        # 从路径列表中读取出图片
        imgs = tf.read_file(input_queue[0])
        labels = input_queue[1]

        # 对图片进行解码
        imgs = tf.image.decode_jpeg(imgs, channels=3)

        # 图片大小统一化
        imgs = tf.image.resize_image_with_crop_or_pad(imgs, img_width, img_height)

        # 生成批次
        # capacity是队列的长度
        imgs_batch, labels_batch = tf.train.batch([imgs, labels],
                                                  batch_size=batch_size,
                                                  num_threads=64,
                                                  capacity=capacity)

        image_batch = tf.cast(imgs_batch, tf.float32)
        label_batch = tf.reshape(labels_batch, [batch_size])

        return image_batch, label_batch


if __name__ == '__main__':

    datas = {'ZJ_img': 'J:/train/sZJ/', 'ZJ_label': 0,
             'CXP_img': 'J:/train/sCXP/', 'CXP_label': 1,
             'WYP_img': 'J:/train/sWYP/', 'WYP_label': 2,
             'ZL_img': 'J:/train/sZL/', 'ZL_label': 3,
             'LZ_img': 'J:/train/sLZ/', 'LZ_label': 4,
             'JML_img': 'J:/train/sJML/', 'JML_label': 5,
             'FCY_img': 'J:/train/sFCY/', 'FCY_label': 6,
             'CP_img': 'J:/train/sCP/', 'CP_label': 7}
    index = [['ZJ_img', 'ZJ_label'],
             ['CXP_img', 'CXP_label'],
             ['WYP_img', 'WYP_label'],
             ['ZL_img', 'ZL_label'],
             ['LZ_img', 'LZ_label'],
             ['JML_img', 'JML_label'],
             ['FCY_img', 'JML_label'],
             ['CP_img', 'CP_label']]
    male_dir = 'J:/train/sZJ/'
    female_dir = 'J:/train/sCXP/'
    data = mydata()
    train, train_label = data.get_datas_dirs_lists(datas,index=index)
    sex = {1: '女', 0: '男'}


    '''for i in range(1000):
        print(sex[train_label[i]])
        img_path = train[i]
        img = cv2.imread(img_path)
        cv2.imshow("image", img)
        cv2.waitKey(2000)
        os.system('cls')'''
    image_batch, label_batch = data.get_batch(5, train, train_label, 64, 64, 10)
    with tf.Session() as sess:
        for img_data in sess.run(image_batch):
            print(sess.run(img_data))
            cv2.imshow("image", sess.run(img_data))
            cv2.waitKey(1000)
    print(image_batch)
