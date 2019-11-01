# -*-coding:utf-8 -*-
import numpy as np
import math
import cv2
import os
import tensorflow as tf
from PIL import Image

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# PATH_OD_PB = 'model/frozen_graph_od.pb'
# PATH_FOVEA_PB = 'model/frozen_graph_fovea.pb'
# WIDTH = 64
# HEIGHT = 64
# PATH_TO_TEST_IMAGES_DIR = 'test_images/'
# PATH_TO_IMAGES_OUTPUT = os.path.join(global_root, 'result/')
# PATH_TO_SAVE_COORDINATE = os.path.join(global_root, 'Coordinate.csv')


class Macular():
    def __init__(self, config):
        self.config = config

    def initial(self):
        graph = tf.get_default_graph()
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(self.config['path_od_pb'], 'rb') as f:
            od_graph_def.ParseFromString(f.read())
            tf.import_graph_def(od_graph_def, name='')

        fovea_graph_def = tf.GraphDef()
        with tf.gfile.GFile(self.config['path_fovea_pb'], 'rb') as f:
            fovea_graph_def.ParseFromString(f.read())
            tf.import_graph_def(fovea_graph_def, name='')

        self.graph = graph
        self.sess = tf.Session(graph=graph)
        return graph

    # def load_image_into_numpy_array(self, image):
    #     (im_width, im_height) = image.shape
    #     return image.reshape((im_height, im_width, 3)).astype(np.uint8)

    def ReadData(self, FilePath, x_center, y_center):
        SrcImage = cv2.imread(FilePath)
        row, column, channel = SrcImage.shape

        BoxX = max(0, round((x_center - 0.3) * column))
        BoxXM = min(column, BoxX + round(column * 0.6))
        BoxW = BoxXM - BoxX
        BoxY = max(0, round((y_center - 0.3) * row))
        BoxYM = min(row, BoxY + round(row * 0.6))
        BoxH = BoxYM - BoxY

        TestImage = SrcImage[int(BoxY):int(
            BoxY + BoxH), int(BoxX):int(BoxX + BoxW)]
        TestImage = cv2.cvtColor(TestImage, cv2.COLOR_BGR2GRAY)
        # clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        # TestImage = clahe.apply(TestImage)
        TestImage = cv2.resize(TestImage, (self.config['width'], self.config['height']))
        TestImage = np.reshape(TestImage, [1, self.config['width'], self.config['height']])
        return TestImage, BoxX, BoxW, BoxY, BoxH, SrcImage

    def detection(self, path):
        graph = self.graph
        sess = self.sess
        with graph.as_default():
            od_image_tensor = graph.get_tensor_by_name('image_tensor:0')
            boxes = graph.get_tensor_by_name('detection_boxes:0')
            fovea_image_tensor = graph.get_tensor_by_name('x:0')
            coordinate = graph.get_tensor_by_name('Sigmoid_5:0')

            with sess.as_default():
                # name = os.path.split(path)[1]
                # name = os.path.splitext(name)[0] + '.csv'
                # coordinate_path = os.path.join(
                #     self.PATH_TO_SAVE_COORDINATE,
                #     name
                # )
                # f = open(coordinate_path, 'w')
                # f.write('FilePath,' + 'X-Macula,' + 'Y-Macula,' + 'R' + '\n')
                try:
                    FilePath = path
                    image_np = cv2.imread(FilePath)
                    image_np_expanded = np.expand_dims(image_np, axis=0)

                    boxes_v = sess.run(boxes, feed_dict={od_image_tensor: image_np_expanded})
                    boxes_np = np.squeeze(boxes_v)[0]
                    box = tuple(boxes_np.tolist())
                    ymin, xmin, ymax, xmax = box

                    x_odc = (xmax + xmin) / 2
                    y_odc = (ymax + ymin) / 2
                    disc_spot = (x_odc, y_odc)
                    width = (xmax - xmin) / 2
                    y_odc_pseudo = (ymax - ymin) / 3.0*2.0 + ymin
                    odd = xmax - xmin

                    # 视盘可能检测出错的情况，给予警告
                    if odd == 0.0 or odd > 0.25:
                        Prompt = 'Warning: Image {} needs to be judged manually again.'.format(path)

                    if x_odc <= 0.5:
                        x_macula = x_odc + odd * 2.5
                        y_macula = y_odc_pseudo
                    else:
                        x_macula = x_odc - odd * 2.5
                        y_macula = y_odc_pseudo
                    if x_macula < 0.05 or x_macula > 0.95:
                        x_macula = 0.5
                        y_macula = 0.6
                    image_data, BoxX, BoxW, BoxY, BoxH, SrcImage = self.ReadData(
                        FilePath, x_macula, y_macula)
                    pred = sess.run(coordinate, feed_dict={
                                    fovea_image_tensor: image_data})
                    x_point = pred[0][0] * BoxW + BoxX
                    y_point = pred[0][1] * BoxH + BoxY
                except:
                    Prompt = 'Warning: Image {} needs to be judged manually again.'.format(path)

                img_name = FilePath.split('/')[-1]
                row, col = image_np.shape[:2]

                # 判断黄斑中心凹距离图像的边缘是否超过2个视盘半径
                if (x_point - 2*odd*col) < 0 or (x_point + 2*odd*col) > col or (y_point - 2*odd*row) < 0 or (y_point + 2*odd*row) > row:
                    flag1 = False
                else:
                    flag1 = True

                # 判断视盘与黄斑中心连线和水平线的夹角是否不大于正负24度
                angle = math.atan2(abs(y_point - y_odc * row), abs(x_point - x_odc * col))
                theta = angle * (180 / math.pi)
                if theta <= 24:
                    flag2 = True
                else:
                    flag2 = False

                # flag=True表示图像质量符合要求，flag=False表示不符合
                flag = (flag1 and flag2)
                cv2.rectangle(SrcImage, (int(xmin*col), int(ymin*row)),
                              (int(xmax*col), int(ymax*row)), (255, 255, 255), 3)
                cv2.circle(SrcImage, (int(x_point), int(
                    y_point)), 10, (255, 255, 255), 5)
                output_path = self.config['path_to_images_output'] + img_name
                cv2.imwrite(output_path, SrcImage)

                R = odd * col / 2.0
                # f.write(FilePath + ',' + str(x_point) + ',' +
                #         str(y_point) + ',' + str(R) + '\n')

                # f.close()

                spot = (int(x_point), int(y_point))
                return output_path, spot, R, theta, disc_spot, width, flag1, flag2

# detection(PATH_TO_TEST_IMAGES_DIR)
