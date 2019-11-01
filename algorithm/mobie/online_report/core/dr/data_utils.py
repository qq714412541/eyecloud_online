import os
import random

import cv2 as cv
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from .retinanet.keras_retinanet import models
from .retinanet.keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from .retinanet.keras_retinanet.utils.visualization import draw_box, draw_caption
from .retinanet.keras_retinanet.utils.colors import label_color

from .configs import network_configs, FLAGS


class DataManager:
    def __init__(self):
        self.config = network_configs[FLAGS['network']]
        self.caches_path = os.path.join(FLAGS['save_path'], 'cache')
        if not os.path.exists(self.caches_path):
            os.mkdir(self.caches_path)
        self.lesion_model = None

    def initial(self, data_dir, batch_size):
        self.batch_size = batch_size
        self.data_dir = data_dir

        self.load_directory(data_dir)
        self.intial_dataset()
        self.build_dataset()

    def load_directory(self, dir):
        self.classes = None
        self.raw_dataset = {
            'train': [[], []],
            'test': [[], []],
            'val': [[], []]
        }

        for cate in ['train', 'test', 'val']:
            cate_path = os.path.join(dir, cate)
            if not os.path.exists(cate_path):
                raise Exception('No train/test/val folders in data dir')

            for root, dirs, files in os.walk(cate_path):
                tail = os.path.split(root)[-1]
                if tail != cate:
                    if tail not in self.classes.keys():
                        raise Exception(
                            'Folders does not have identical classes')
                    else:
                        paths = [os.path.join(root, f) for f in files]
                        labels = [self.classes[tail]] * len(paths)
                        self.raw_dataset[cate][0] += paths
                        self.raw_dataset[cate][1] += [self.classes[tail]] * len(paths)
                else:
                    self.classes = {
                        class_: i for i, class_ in enumerate(dirs)
                    }
                    self.num_classes = len(self.classes)

        # shuffle train data
        datas = list(zip(self.raw_dataset['train'][0], self.raw_dataset['train'][1]))
        random.shuffle(datas)
        paths = [data[0] for data in datas]
        labels = [data[1] for data in datas]
        self.raw_dataset['train'][0] = paths
        self.raw_dataset['train'][1] = labels

    def build_dataset(self):
        config = self.config
        caches_path = self.caches_path

        def _parse_function(filepath):
            if not self.lesion_model:
                self.lesion_model = models.load_model(FLAGS['lesion_model_path'], backbone_name='resnet50')
            return stack_with_lesion(config, filepath, self.lesion_model)

        def _build_tfrecords(raw_dataset, cate):
            cache_name = cate + '.tfrecords'
            cache_path = os.path.join(caches_path, cache_name)
            if cache_name not in os.listdir(caches_path):
                print('creating TFRecords in: {}'.format(cache_path))
                writer = tf.python_io.TFRecordWriter(cache_path)
                samples = raw_dataset[0]
                labels = raw_dataset[1]
                for i in tqdm(range(len(samples))):
                    filepath = samples[i]
                    integrated_image = _parse_function(filepath).tostring()
                    example = tf.train.Example(features=tf.train.Features(
                        feature={
                            'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[integrated_image])),
                            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[i]]))
                        }))
                    writer.write(example.SerializeToString())
                writer.close()
            return cache_path

        self.train_tfrecords_path = _build_tfrecords(self.raw_dataset['train'], 'train')
        self.test_tfrecords_path = _build_tfrecords(self.raw_dataset['test'], 'test')
        self.val_tfrecords_path = _build_tfrecords(self.raw_dataset['val'], 'val')

    def intial_dataset(self):
        config = self.config

        def _parse_record(example):
            features = {
                'img': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)
            }
            parsed_features = tf.parse_single_example(example, features=features)
            image = tf.decode_raw(parsed_features['img'], tf.float64)
            image = tf.cast(image, tf.float32)
            image = tf.reshape(
                image,
                shape=[config['INPUT_HEIGHT'], config['INPUT_WIDTH'], 5]
            )
            label = tf.cast(parsed_features['label'], tf.int32)
            label = tf.one_hot(label, depth=self.num_classes, on_value=1)

            return image, label

        self.tfrecords_path = tf.placeholder(tf.string, shape=())
        dataset = tf.data.TFRecordDataset(self.tfrecords_path)
        dataset = dataset.shuffle(FLAGS['shuffle_buff'])
        dataset = dataset.map(_parse_record, num_parallel_calls=FLAGS['num_preprocess_parallel'])
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(FLAGS['prefetch_size'])
        self.iterator = dataset.make_initializable_iterator()

    def switch_to_train_dataset(self, sess):
        sess.run(self.iterator.initializer, feed_dict={
            self.tfrecords_path: self.train_tfrecords_path
        })
        self._update_dataset_state('train')

    def switch_to_test_dataset(self, sess):
        sess.run(self.iterator.initializer, feed_dict={
            self.tfrecords_path: self.test_tfrecords_path
        })
        self._update_dataset_state('test')

    def switch_to_val_dataset(self, sess):
        sess.run(self.iterator.initializer, feed_dict={
            self.tfrecords_path: self.val_tfrecords_path
        })
        self._update_dataset_state('val')

    def _update_dataset_state(self, cate):
        self.samples_num = len(self.raw_dataset[cate][0])
        self.steps = self.samples_num // self.batch_size + 1
        self.sizes = [self.batch_size] * self.steps
        self.sizes[-1] = self.samples_num % self.batch_size

    def next_batch(self):
        return self.iterator.get_next()


def stack_with_lesion(config, filepath, lesion_model, lesion_nums=2):
    image = read_image_bgr(filepath)
    image, _ = resize_image(image)
    image_feed = preprocess_image(image, mode='caffe')
    image_feed = np.expand_dims(image_feed, axis=0)

    # generate lesion label
    lesion_result = lesion_model.predict_on_batch(image_feed)
    lesion_labels = generate_lesion_labels(
        lesion_result,
        image_feed.shape[1:3],
        lesion_nums
    )

    return stack(config, image, lesion_labels, lesion_nums)


def generate_lesion_labels(lesion_result, label_size, lesion_nums, lesion_types):
    boxes, scores, labels = lesion_result

    lesion_labels = [np.zeros(label_size) + 1e-4 for _ in range(lesion_nums)]
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        x1, y1, x2, y2 = box.astype(int)
        if label in lesion_types:
            lesion_label = lesion_labels[label]
            lesion_label[y1:y2, x1:x2] = np.maximum(lesion_label[y1:y2, x1:x2], score)
    print(len(lesion_labels))

    return lesion_labels


def stack(config, image, lesion_labels, lesion_nums=2):
    # stack lesion label
    lesion_labels = [cv.resize(
        lesion_label,
        dsize=(config['INPUT_HEIGHT'], config['INPUT_WIDTH']),
        interpolation=cv.INTER_LINEAR
    ) for lesion_label in lesion_labels]
    stacked_lesion_label = np.stack(lesion_labels, axis=-1)

    # preprocess image
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(
        image,
        dsize=(config['INPUT_HEIGHT'], config['INPUT_WIDTH']),
        interpolation=cv.INTER_LINEAR
    )
    image = preprocess_image(image, mode='tf')

    stacked_image = np.concatenate((image, stacked_lesion_label), axis=-1)
    assert stacked_image.shape[-1] == 3 + lesion_nums
    return stacked_image
