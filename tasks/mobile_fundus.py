# -*- coding: utf-8 -*-

from tasks.celery_config import celery
from celery.result import AsyncResult
import os, json
from services.logger import LogService
import tensorflow as tf
import numpy as np
import requests, datetime
import time as TIME
# from algorithms.fundus.mobile.FundusAnalysis import FundusAnalysis
from data_access.sqlforZhaoqing import AppZhaoqing
from algorithms.mobile.online_report.FundusAnalysis import FundusAnalysis

import random

logger = LogService()
sqlzq = AppZhaoqing()

# filePath = "2018/8/26/c0de15f2b75f5f8098349299867b2ea9-1535291315852.jpeg"
# DOWNLOAD_PATH = './download'
# BACKEND_HOST = 'localhost:5000'
# BACKEND_NOTIFY_URL = 'http://' + BACKEND_HOST + '/api/notify/fundusImage'
CONFIG_PATH = './algorithms/mobile/online_report/config.py'
MODEL_DIR = './algorithms/mobile/online_report/models'
RECORD_DIR = './algorithms/mobile/online_report'

CALL_CELERY = False
fundusAnalysis = None
sqlurl = 'http://172.18.4.235:5001/upload'  # sqlserver

@celery.task
def dr_classify(app_id, time, sign, original_path, exam_data):
    global fundusAnalysis, CALL_CELERY
    print(CALL_CELERY)

    if not CALL_CELERY:
        CALL_CELERY = True
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >./tmp')
        memory_gpu = [int(x.split()[2]) for x in open('./tmp', 'r').readlines()]
        print(memory_gpu)
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            os.environ['CUDA_VISIBLE_DEVICES'] = str(os.getenv('ENV_PORT'))
            print('now use', str(os.getenv('ENV_PORT')))
            fundusAnalysis = FundusAnalysis()
            fundusAnalysis.initial(record_dir=RECORD_DIR, model_dir=MODEL_DIR)
            os.system('rm tmp')

    #reportpath = list(range(len(original_path)))
    #abnormal = list(range(len(original_path)))
    reportid = list(range(len(original_path)))
    #aitime = list(range(len(original_path)))
    #curestatus = list(range(len(original_path)))
    #eye = list(range(len(original_path)))
    sickname = list(range(len(original_path)))

    for i in range(len(original_path)):
        reportid[i] = str(exam_data['examId']) + '_%s' % (str(i))
        sickname[i] = str(exam_data['sickName'])


    time_start = TIME.time()

    report, _, _ = fundusAnalysis.analysis(
        original_path, sickname, reportid
    )

    time_end = TIME.time()
    cost = time_end - time_start
    print(report)

    timestr = upload_mobile_report(exam_data, report, cost)

    return timestr


def upload_mobile_report(exam_data, report, time_cost):
    for i in range(len(report)):   #还有图像的话可以在这里加

        if report[i]['report_path']:

            files = {'file': open(report[i]['report_path'], 'rb')}
            res = requests.post(url=sqlurl, files=files)
            print(res.text)
            print(res)
            report[i]['report_path'] = json.loads(res.text)['reportPath']

        if report[i]['output_images']['retinal_vessel_image']:

            files = {'file': open(report[i]['output_images']['retinal_vessel_image'], 'rb')}   #血管分割
            res = requests.post(url=sqlurl, files=files)
            print(res.text)
            print(res)
            report[i]['output_images']['retinal_vessel_image'] = json.loads(res.text)['reportPath']

        if report[i]['output_images']['quadrant_segmentation_image']:

            files = {'file': open(report[i]['output_images']['quadrant_segmentation_image'], 'rb')}  # 血管分析ROI
            res = requests.post(url=sqlurl, files=files)
            print(res.text)
            print(res)
            report[i]['output_images']['quadrant_segmentation_image'] = json.loads(res.text)['reportPath']

        if report[i]['output_images']['macular_image']:

            files = {'file': open(report[i]['output_images']['macular_image'], 'rb')}  # 黄斑中心定位以及视盘检测
            res = requests.post(url=sqlurl, files=files)
            print(res.text)
            print(res)
            report[i]['output_images']['macular_image'] = json.loads(res.text)['reportPath']

        if report[i]['output_images']['exudation_image']:

            files = {'file': open(report[i]['output_images']['exudation_image'], 'rb')}  # 渗出位置检测
            res = requests.post(url=sqlurl, files=files)
            print(res.text)
            print(res)
            report[i]['output_images']['exudation_image'] = json.loads(res.text)['reportPath']

        if report[i]['output_images']['bleed_image']:

            files = {'file': open(report[i]['output_images']['bleed_image'], 'rb')}  # 出血
            res = requests.post(url=sqlurl, files=files)
            print(res.text)
            print(res)
            report[i]['output_images']['bleed_image'] = json.loads(res.text)['reportPath']

        if report[i]['output_images']['optic_image']:
            files = {'file': open(report[i]['output_images']['optic_image'], 'rb')}  # 视杯分割
            res = requests.post(url=sqlurl, files=files)
            print(res.text)
            print(res)
            report[i]['output_images']['optic_image'] = json.loads(res.text)['reportPath']

        if report[i]['output_images']['microaneurysms_image']:
            files = {'file': open(report[i]['output_images']['microaneurysms_image'], 'rb')}  # 微血管瘤
            res = requests.post(url=sqlurl, files=files)
            print(res.text)
            print(res)
            report[i]['output_images']['microaneurysms_image'] = json.loads(res.text)['reportPath']


    '''data = {
        'report' : report,
        'reportid' : reportid,
        'original_path' : original_path
    }'''
    print(report)

    data_json = json.dumps(report, ensure_ascii=False)

    timestr = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')



    sqlzq.aftersqlZhaoqing(exam_data['examId'],  timestr, 1, data_json)

    ###########################

    return timestr



