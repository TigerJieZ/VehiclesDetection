import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import input_data
import model
import time

class iDD:
    def __init__(self, save_path='J:/checkpoint/'):
        self.data = input_data.mydata()
        self.model = model.model()
        self.train = model.train()

    def get_one_image(self, dir):
        n = len(dir)
        lnd = np.random.randint(0, n)
        img_dir = dir[lnd]

        image = Image.open(img_dir)
        print("img_dir", img_dir)
        mat = cv2.imread(img_dir)
        cv2.imshow("img", mat)
        cv2.waitKey(1000 * 1)
        a=input()
        image = image.resize([112, 92])
        image = np.array(image)
        return image

    def evaluate_one_image(self, datas,index):
        # datas = {'ZJ_img': 'J:/train/sZJ/', 'ZJ_label': 0,
        #          'CXP_img': 'J:/train/sCXP/', 'CXP_label': 1,
        #          'WYP_img': 'J:/train/sWYP/', 'WYP_label': 2,
        #          'ZL_img': 'J:/train/sZL/', 'ZL_label': 3,
        #          'LZ_img': 'J:/train/sLZ/', 'LZ_label': 4,
        #          'JML_img': 'J:/train/sJML/', 'JML_label': 5,
        #          'FCY_img': 'J:/train/sFCY/', 'FCY_label': 6,
        #          'CP_img': 'J:/train/sCP/', 'CP_label': 7}
        # index = [['ZJ_img', 'ZJ_label'],
        #          ['CXP_img', 'CXP_label'],
        #          ['WYP_img', 'WYP_label'],
        #          ['ZL_img', 'ZL_label'],
        #          ['LZ_img', 'LZ_label'],
        #          ['JML_img', 'JML_label'],
        #          ['FCY_img', 'JML_label'],
        #          ['CP_img', 'CP_label']]
        train, train_label = self.data.get_datas_dirs_lists(datas,index)

        image_array = self.get_one_image(train)
        start_time = time.time()
        with tf.Graph().as_default():
            BATCH_SIZE = 1
            N_CLASSES = 8

            image = tf.cast(image_array, tf.float32)
            image_predit = tf.reshape(image, [92, 112, 1])
            image = tf.reshape(image, [1, 92, 112, 1])

            logit = self.model.inference(image, BATCH_SIZE, N_CLASSES)
            logit = tf.nn.softmax(logit)

            logs_train_dir = 'J:/checkpoint/'
            saver = tf.train.Saver()
            x = tf.placeholder(tf.float32, shape=[92, 112, 1])
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(logs_train_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    print('No checkpoint file found')
                prediction = sess.run(logit, feed_dict={x: sess.run(image_predit)})
                print(time.time() - start_time)
                max_index = np.argmax(prediction)
                if max_index == 3:
                    print("This is a ZL whith possibility ", prediction[:, 3][0]*100,'%')
                elif max_index==7:
                    print("This is a CP whith possibility \n", prediction[:, 7][0]*100,'%')
        return max_index


def cpp(dir=""):
    IDD=iDD()
    return IDD.evaluate_one_image([dir])


if __name__ == '__main__':
    IDD = iDD()
    datas = {'ZJ_img': 'J:/test_ZJ_CXP/', 'ZJ_label': 0,}
    index = [['ZJ_img', 'ZJ_label'],]
    for i in range(10):
        IDD.evaluate_one_image(datas,index)
   # cpp("J:/Visual Studio 2015/kinect_exp/IdentityDetection/x64/Debug/img/")
