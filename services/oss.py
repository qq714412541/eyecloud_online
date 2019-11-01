# -*- coding: utf-8 -*-

import os, json
import oss2, time

CONFIG_PATH = './config/oss.json'
with open(CONFIG_PATH, 'r') as config_file:
    config = json.load(config_file)
    ACCESSKEY_ID = config['accessKeyID']
    ACCESSKEY_SECRET = config['accessKeySecret']
    HOST = config['host']
    BUCKET = config['bucket']
    DOWNLOAD_DIR = config['downloadPath']
    UPLOAD_DIR = config['uploadPath']

class OSSService():

    def __init__(self):
        auth = oss2.Auth(ACCESSKEY_ID, ACCESSKEY_SECRET)
        self.bucket = oss2.Bucket(auth, HOST, BUCKET)
        if not os.path.exists(DOWNLOAD_DIR):
            os.makedirs(DOWNLOAD_DIR)
        if not os.path.exists(UPLOAD_DIR):
            os.makedirs(UPLOAD_DIR)

    def download_file(self, object_path):
        store_dir = os.path.join(
            DOWNLOAD_DIR,
            self.get_dir_from_file_path(object_path)
        )

        if not os.path.exists(store_dir):
            os.makedirs(store_dir)

        store_path = os.path.join(DOWNLOAD_DIR, object_path)
        self.bucket.get_object_to_file(object_path, store_path)

        return store_path

    def get_dir_from_file_path(self, file_path):
        if file_path.find('/') == -1:
            return ''
        else:
            return file_path[0:file_path.rindex('/')]

    def upload_file(self, file_path, object_path):
        file_name = str(time.time())
        object_path = UPLOAD_DIR + '/' + file_name + '.jpg'
        self.bucket.put_object_from_file(object_path, file_path)
        return object_path