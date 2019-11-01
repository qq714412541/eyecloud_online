# -*- coding: utf-8 -*-
from tasks.celery_config import app
from celery.result import AsyncResult
import os, json
import tensorflow as tf
import numpy as np
import requests
import random
import pprint
import time as TIME
import datetime

from data_access.sqlforEyecloud import AppEyecloud
from data_access.sqlforZhaoqing import AppZhaoqing
from services.logger import LogService
from services.oss import OSSService
from algorithms.mobile.online_report.FundusAnalysis import FundusAnalysis

ossservice = OSSService()
logger = LogService()

UPLOAD_URL = 'http://172.18.4.235:5001/upload'
MODEL_DIR = './algorithms/mobile/online_report/models'
RECORD_DIR = './algorithms/mobile/online_report'

fundus_analysis = None

def get_fundus_analysis():
    global fundus_analysis
    print(' Fundus initial status: %r' % (bool(fundus_analysis)))

    if not fundus_analysis:

        os.system('nvidia-smi -q -d Memory | grep -A4 GPU| grep Free >./tmp')
        memory_gpu = [int(x.split()[2]) for x in open('./tmp', 'r').readlines()]
        print(memory_gpu)
        #with tf.variable_scope('fundus', reuse=tf.AUTO_REUSE):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(os.getenv('ENV_PORT'))
        print(' Now use: %s' % (os.getenv('ENV_PORT')))
        fundus_analysis = FundusAnalysis()
        fundus_analysis.initial(record_dir=RECORD_DIR, model_dir=MODEL_DIR)
        os.system('rm tmp')

    return fundus_analysis


@app.task
def online_fundus_analysis(id_, images_path, callback):

    store_paths = list(range(len(images_path)))
    ids = list(range(len(images_path)))
    report_ids = list(range(len(images_path)))


    for index in range(0, len(images_path)):
        store_paths[index] = ossservice.download_file(images_path[index])
        ids[index] = str(id_)
        report_ids[index] = '%d_%d' % (id_, index)

    reports, _, _ = get_fundus_analysis().analysis(
        store_paths, ids, report_ids
    )

    print(reports)

    f_status_list = process_online_report(id_, store_paths, reports, callback)

    return f_status_list

def process_online_report(id_, store_paths, reports, callback):
    assert(len(reports) == len(store_paths))
    results = []
    f_status_list = []

    for index in range(len(reports)):
        if os.path.exists(store_paths[index]):
            os.remove(store_paths[index])

        f_status = reports[index]['status']
        f_status_list.append(f_status)

        if f_status == 1:  # 非眼底照
            result = {
                'f_status': f_status
            }
            results.append(result)
            continue

        # 2-1 黄斑图
        macular_image = ossservice.upload_file(
            reports[index]['output_images']['macular_image'], ''
        )
        # 2-2 出血范围图
        bleed_image = ossservice.upload_file(
            reports[index]['output_images']['bleed_image'], ''
        )
        # 2-3 出血直方图
        bleed_histogram = ossservice.upload_file(
            reports[index]['output_images']['bleed_histogram'], ''
        )
        # 2-4 渗出范围图
        exudation_image = ossservice.upload_file(
                reports[index]['output_images']['exudation_image'], ''
        )
        # 2-5 渗出直方图
        exudation_histogram = ossservice.upload_file(
            reports[index]['output_images']['exudation_histogram'], ''
        )

        result = {
            'report': [reports[index]],
            'macular_image': macular_image,
            'bleed_image': bleed_image,
            'bleed_histogram': bleed_histogram,
            'exudation_image': exudation_image,
            'exudation_histogram': exudation_histogram,
            'f_status': f_status
        }

        if f_status == 0:   # 视盘清晰再取血管分析结果
            # 3-1 血管分割图
            retinal_vessel_image = ossservice.upload_file(
                reports[index]['output_images']['retinal_vessel_image'], ''
            )
            # 3-2 象限分割图
            quadrant_segmentation_image = ossservice.upload_file(
                reports[index]['output_images']['quadrant_segmentation_image'], ''
            )
            # 3-3 血管分析-患者血管长度直方图
            a_patient_length_histogram = ossservice.upload_file(
                reports[index]['output_images']['a-patient_length_histogram'], ''
            )
            # 3-5 血管分析-患者血管直径直方图
            a_patient_diameter_histogram = ossservice.upload_file(
                reports[index]['output_images']['a-patient_diameter_histogram'], ''
            )

            result['retinal_vessel_image'] = retinal_vessel_image
            result['quadrant_segmentation_image'] = quadrant_segmentation_image
            result['a-patient_length_histogram'] = a_patient_length_histogram
            result['a-patient_diameter_histogram'] = a_patient_diameter_histogram

        reports[index].pop('output_images')
        reports[index].pop('a-normal_diameter_histogram')
        reports[index].pop('a-normal_length_histogram')
        reports[index].pop('watermark1')
        reports[index].pop('watermark2')
        reports[index].pop('watermark3')
        reports[index].pop('watermark4')
        reports[index].pop('template1')
        reports[index].pop('template2')
        reports[index].pop('template3')
        reports[index].pop('template4')

        results.append(result)

    print(results)

    results_json = json.dumps(results, ensure_ascii=False)

    # Save status and result to database

    connect = AppEyecloud()
    connect.aftersqlEyecloud(results_json, id_, '1', '1')

    # Notify the backend
    request_body = {
        'id': id_,
        'results': results
    }

    res = requests.post(url=callback, json=request_body, headers={'Content-Type': 'application/json;charset=utf-8'})
    print(res.text)
    print(res)

    return f_status

@app.task
def mobile_fundus_analysis(app_id, time, sign, original_paths, exam_data):

    report_ids = list(range(len(original_paths)))
    sick_names = list(range(len(original_paths)))

    for index in range(len(original_paths)):
        report_ids[index] = str(exam_data['examId']) + '_%s' % (str(index))
        sick_names[index] = str(exam_data['sickName'])

    time_start = TIME.time()

    reports, _, _ = get_fundus_analysis().analysis(
        original_paths, sick_names, report_ids
    )

    time_end = TIME.time()
    time_cost = time_end - time_start

    timestr = process_mobile_report(exam_data, reports, time_cost)

    return timestr

def process_mobile_report(exam_data, reports, time_cost):
    for index in range(len(reports)):   #还有图像的话可以在这里加

        if reports[index]['report_path']:

            files = {'file': open(reports[index]['report_path'], 'rb')}
            res = requests.post(url=UPLOAD_URL, files=files)
            print(res.text)
            print(res)
            reports[index]['report_path'] = json.loads(res.text)['reportPath']

        if reports[index]['output_images']['retinal_vessel_image']:

            files = {'file': open(reports[index]['output_images']['retinal_vessel_image'], 'rb')}   #血管分割
            res = requests.post(url=UPLOAD_URL, files=files)
            print(res.text)
            print(res)
            reports[index]['output_images']['retinal_vessel_image'] = json.loads(res.text)['reportPath']

        if reports[index]['output_images']['quadrant_segmentation_image']:

            files = {'file': open(reports[index]['output_images']['quadrant_segmentation_image'], 'rb')}  # 血管分析ROI
            res = requests.post(url=UPLOAD_URL, files=files)
            print(res.text)
            print(res)
            reports[index]['output_images']['quadrant_segmentation_image'] = json.loads(res.text)['reportPath']

        if reports[index]['output_images']['macular_image']:

            files = {'file': open(reports[index]['output_images']['macular_image'], 'rb')}  # 黄斑中心定位以及视盘检测
            res = requests.post(url=UPLOAD_URL, files=files)
            print(res.text)
            print(res)
            reports[index]['output_images']['macular_image'] = json.loads(res.text)['reportPath']

        if reports[index]['output_images']['exudation_image']:

            files = {'file': open(reports[index]['output_images']['exudation_image'], 'rb')}  # 渗出位置检测
            res = requests.post(url=UPLOAD_URL, files=files)
            print(res.text)
            print(res)
            reports[index]['output_images']['exudation_image'] = json.loads(res.text)['reportPath']

        if reports[index]['output_images']['bleed_image']:

            files = {'file': open(reports[index]['output_images']['bleed_image'], 'rb')}  # 出血
            res = requests.post(url=UPLOAD_URL, files=files)
            print(res.text)
            print(res)
            reports[index]['output_images']['bleed_image'] = json.loads(res.text)['reportPath']

        if reports[index]['output_images']['optic_image']:
            files = {'file': open(reports[index]['output_images']['optic_image'], 'rb')}  # 视杯分割
            res = requests.post(url=UPLOAD_URL, files=files)
            print(res.text)
            print(res)
            reports[index]['output_images']['optic_image'] = json.loads(res.text)['reportPath']

        if reports[index]['output_images']['microaneurysms_image']:
            files = {'file': open(reports[index]['output_images']['microaneurysms_image'], 'rb')}  # 微血管瘤
            res = requests.post(url=UPLOAD_URL, files=files)
            print(res.text)
            print(res)
            reports[index]['output_images']['microaneurysms_image'] = json.loads(res.text)['reportPath']


    '''data = {
        'report' : report,
        'reportid' : reportid,
        'original_path' : original_path
    }'''
    print(reports)

    data_json = json.dumps(reports, ensure_ascii=False)

    timestr = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


    sqlzq = AppZhaoqing()
    sqlzq.aftersqlZhaoqing(exam_data['examId'],  timestr, 1, data_json)

    ###########################

    return timestr

