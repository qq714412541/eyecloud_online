# -*- coding: utf-8 -*-

import urllib, os, json
import traceback,datetime
from flask import Flask, request, redirect, jsonify, Blueprint
from services.logger import LogService
#from services.download import DownloadService
from services.check import CheckService
#from services.app import AppService
from services.encryption import EncrptionService
from data_access.sqlforEyecloud import AppEyecloud
from data_access.sqlforZhaoqing import AppZhaoqing

from services.fundus_image import FundusImageService
import tasks.surface_image as surfaceImage
import os, time, hashlib

commit_api = Blueprint('commit', __name__)

logger = LogService()
#download_service = DownloadService()
#app_service = AppService()
fundus_image_service = FundusImageService()

@commit_api.route('/postExam', methods=['POST'])
def commit_post_exam():
    data = request.get_json()
    logger.info(data)
    response = {}

    status = 200

    try:
        if not CheckService.check_json_key(
            data=data,
            key_dict={
                'appId': 'str', 'timeStr': 'str',
                'signStr': 'str', 'data': 'dict'
            }
        ):

            response = {
                'code': 10001
            }
            return jsonify(response), status

        elif not CheckService.check_json_key(
            data=data['data'],
            key_dict={
                'examId': 'str', 'sickName': 'str', 'sickSource': 'str', 'sickSex':'str',
                'noteStr': 'str','image': 'list', 'sickAge':'str'
            }
        ):

            response = {
                'code': 10002
            }
            return jsonify(response), status

        else:
            app_id = data['appId']
            time = data['timeStr']
            sign = data['signStr']
            exam_data = data['data']

            ## do sth.
            app_key =  'e10adc3949ba59abbe56e057f20f883e'
            print(app_key)
            if len(app_key) == 0:
                response = {
                    'code': 10003
                }
                return jsonify(response), status
            else:
                rep = int
                connect = AppZhaoqing()
                rep = connect.beforesqlZhaoqing(app_id, app_key, time, exam_data, 0)

                if rep == 0:
                    response = {
                        'code': 10004
                    }
                    return jsonify(response), status
                else:
                    # add task to the queue
                    #fundusImage.dr_classify.delay(id_, image, callback)


                    signstr = EncrptionService.sign_data(exam_data, app_key)#check signStr
                    print(signstr)

                    timestr =datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    fundus_image_service.run_mobile_fundus(app_id, time, sign, exam_data)




                    response = {
                    'status': 200,
                    'code': 1,
                    #'msg': '',  not need
                    'timeStr': timestr,
                    'signStr': signstr
                        }
                    return jsonify(response), status

    except Exception as err:
        status = 500
        logger.error(traceback.format_exc())

    return jsonify(response), status




###############
#can be ignored
###############

###############
#can be ignored
###############




@commit_api.route('/surfaceImage', methods=['POST'])
def commit_surface_image():
    rep=int
    data = request.get_json()
    logger.info(data)

    response = {}
    status = 200
    service_flag = False

    # 1.Check the data format
    if 'id' in data and 'images' in data and 'genre' in data and 'callback' in data:
        if isinstance(data['id'], int):
            service_flag = True

    if not service_flag:

        response = {
            'code': 100021
        }
        return jsonify(response), status
    id_ = data['id']
    images = data['images']
    image = images[0]
    genre = data['genre']
    callback=data['callback']

    try:
        if genre ==2:
            connect = AppEyecloud()
            rep= connect.beforesqlEyecloud(id_, '0', genre,rep)

            if rep == 0:
                response = {
                    'code': 100022
                }

                return jsonify(response), status
            else:
            # add task to the queue
                surfaceImage.surface_image.delay(id_, image,callback)
                response = {
                'code': 2
                }
                return jsonify(response), status



        else:
            response = {
            'code': 100023
            }

            return jsonify(response), status



    except Exception as err:
        response = {
            'code': 100024
        }
        status = 500
        logger.error(traceback.format_exc())

    return jsonify(response), status




@commit_api.route('/fundusImage', methods=['POST'])
def commit_fundus_image():
    rep=int
    data = request.get_json()
    logger.info(data)

    response = {}
    status = 200
    service_flag = False

    # 1.Check the data format
    if 'id' in data and 'images' in data and 'genre' in data and 'callback' in data:
        if isinstance(data['id'], int):
            service_flag = True

    if not service_flag:
        response = {
            'code': 100021
        }
        return jsonify(response), status

    id_ = data['id']
    image = data['images']

    genre = data['genre']
    callback=data['callback']

    try:
        if genre == 1:
            connect = AppEyecloud()
            rep= connect.beforesqlEyecloud(id_, '0', genre,rep)

            if rep == 0:
                response = {
                    'code': 100022
                }
                return jsonify(response), status
            else:
                # add task to the queue
                fundus_image_service.run_online_fundus(id_, image, callback)
                response = {
                    'code': 1
                }
                return jsonify(response), status



        else:
            response = {
                'code': 100023
            }
            return jsonify(response), status



    except Exception as err:
        response = {
            'code': 100024
        }
        status = 500
        logger.error(traceback.format_exc())


    return jsonify(response), status
