# -*- coding: utf-8 -*-
from tasks.celery_config import app
from celery.result import AsyncResult
from services.logger import LogService
from services.oss import OSSService

import tensorflow as tf
import numpy as np
import os, requests
import json

from data_access.sqlforEyecloud import AppEyecloud
import random

ossservice = OSSService()
#download_service = ossservice.download_file()
logger = LogService()
#upload_service = ossservice.upload_file()


# BACKEND_HOST = 'localhost:5000'
# BACKEND_NOTIFY_URL = 'http://' + BACKEND_HOST + '/api/notify/surfaceImage'

from algorithms.surface.inference import Inference

surface_analysis = None

def get_surface_analysis():
    global surface_analysis
    print(' Surface initial status: %r' % (bool(surface_analysis)))

    if not surface_analysis:
        os.system('nvidia-smi -q -d Memory | grep -A4 GPU| grep Free >./tmp')
        memory_gpu = [int(x.split()[2]) for x in open('./tmp', 'r').readlines()]
        print(memory_gpu)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(os.getenv('ENV_PORT'))
        print(' Now use: %s' % (os.getenv('ENV_PORT')))
        surface_analysis = Inference()
        os.system('rm tmp')

    return surface_analysis

@app.task
def surface_image(id_, image_path, callback):

    results = []
    storge_path = ossservice.download_file(image_path)
    print(storge_path)

    output_path = get_surface_analysis().predict(storge_path, './output')
    object_path = str()
    surface_path = ossservice.upload_file(output_path,object_path)

    report=[]
    list=[('report_time','2019'),('status',0)]
    dic = dict(list)
    report = [dic]

    s_status = report[0]['status']

    if s_status==0:
        list2=[('cornea_area','200'),('corneal_ulcer','有'),('ulcer_stage','3'),('ulcer_type','点片混合'),
               ('ulcer_location','第一、二象限'),('ulcer_area','70'),('ulcer_percentage','35%')]
        dic2 = dict(list2)
        report[0].update(dic2)

        '''
        report[0]['cornea_area'] = '200'
        report[0]['corneal_ulcer'] = '有'
        report[0]['ulcer_stage'] = '3'
        report[0]['ulcer_type'] = '点片混合'
        report[0]['ulcer_location'] = '第一、二象限'
        report[0]['ulcer_area'] = '70'
        report[0]['ulcer_percentage'] = '35%'
        '''

    elif s_status==1:
        list2=[('cornea_area','200'),('corneal_ulcer','无')]
        dic2=dict(list2)
        report[0].update(dic2)

        '''
        report[0]['cornea_area'] = '200'
        report[0]['corneal_ulcer'] = '无'
        '''

    elif s_status==2:
        s_status=2


    result = {
        'report' : report,
        'surface_path':surface_path,
        's_status' : s_status,

    }
    results.append(result)


    #output_path = upload_service.upload_file_from_oss()
    # 1. Run the algorithm
    '''
    level, cornealArea, damagedArea = surface_example(image_path)

    results = []
    result = {
        'level': level,
        'cornealArea': cornealArea,
        'damagedArea': damagedArea

    }
    results.append(result)
    '''


    results_json = json.dumps(results, ensure_ascii=False)  # 转json

    connect = AppEyecloud()  # 实例化
    connect.aftersqlEyecloud(results_json, id_, '1', '2')  # 输入参数

    # 2. Save status and result to database

    # 3. Notify the backend
    request_body = {
        'id': id_,
        'results': results
    }

    res = requests.post(url=callback, json=request_body, headers={'Content-Type': 'application/json;charset=utf-8'})
    print(res.text)
    print(res)

    return id_, surface_path





