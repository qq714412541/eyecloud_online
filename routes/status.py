# -*- coding: utf-8 -*-

from flask import Flask, request, redirect, jsonify, Blueprint
from services.logger import LogService
#from services.download import DownloadService
from services.check import CheckService
from data_access.sqlforEyecloud import AppEyecloud
from data_access.sqlforZhaoqing import AppZhaoqing
#from services.app import AppService
from services.encryption import EncrptionService
import traceback,datetime

logger = LogService()
appeyecloud = AppEyecloud()
appzhaoqing = AppZhaoqing()
#app_service = AppService()
status_api = Blueprint('status', __name__)

@status_api.route('/mobileFundus', methods=['POST'])
def getMobileFundusStatus():
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
                    'examId': 'str'
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
            app_key = 'e10adc3949ba59abbe56e057f20f883e'
            print(app_key)
            if len(app_key) == 0:
                response = {
                    'code': 10003
                }
                return jsonify(response), status
            else:
                res = {}


                res = appzhaoqing.checkforZhaoqing(exam_data['examId'])

                print(res)

                signstr = EncrptionService.sign_data(exam_data, app_key)  # check signStr
                timestr = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(signstr)



                response = {
                        'status': 200,

                        'msg': res,
                        'timeStr': timestr,
                        'signStr': signstr
                    }
                return jsonify(response), status

    except Exception as err:
        status = 500
        logger.error(traceback.format_exc())

    return jsonify(response), status





    '''
    response = {
        'code': 10001,
        'error': 'wrong format'
    }
    response = {
        'code': 10002,
        'error': 'no task'
    }
    response = { # pending
        'code': 0,
        'status': 0
    }
    response = { # finished
        'code': 0,
        'status': 1,
        'results': [{
            'type': type_,
            'probability': probability,
            'direction': direction
        }]
    }
    '''





@status_api.route('/fundusImage', methods=['POST'])
def getFundusImageStatus():
    data = request.get_json()
    logger.info(data)
    response = {}

    status = 200

    try:
        if not CheckService.check_json_key(
                data=data,
                key_dict={
                    'id': 'int',
                    'genre': 'int',
                }
        ):
            response = {
                'code': 10001
            }

        else:
            id_ = data['id']
            genre = data['genre']
            # Get algorithm status from database
            #connect = pyc()
            response = appeyecloud.checkforEyecloud(id_, genre, response)

    except Exception as err:
        status = 500
        logger.error(traceback.format_exc())

    '''
    response = {
        'code': 10001,
        'error': 'wrong format'
    }
    response = {
        'code': 10002,
        'error': 'no task'
    }
    response = { # pending
        'code': 0,
        'status': 0
    }
    response = { # finished
        'code': 0,
        'status': 1,
        'results': [{
            'type': type_,
            'probability': probability,
            'direction': direction
        }]
    }
    '''

    return jsonify(response), status


@status_api.route('/surfaceImage', methods=['POST'])
def getSurfaceImageStatus():
    data = request.get_json()
    logger.info(data)
    response = {}

    status = 200

    try:

        if not CheckService.check_json_key(
                data=data,
                key_dict={
                    'id': 'int',
                    'genre': 'int',
                }
        ):
            response = {
                'code': 10001
            }

        else:
            id_ = data['id']
            genre = data['genre']
            # Get algorithm status from database
            #connect = pyc()
            response = appeyecloud.checkforEyecloud(id_, genre, response)


    except Exception as err:
        status = 500
        logger.error(traceback.format_exc())

    '''
    response = {
        'code': 10001,
        'error': 'wrong format'
    }
    response = {
        'code': 10002,
        'error': 'no task'
    }
    response = { # pending
        'code': 0,
        'status': 0
    }
    response = { # finished
        'code': 0,
        'status': 1,
        'result': {
            'level': level,
            'cornealArea': cornealArea,
            'demagedArea': demagedArea
        }
    }
    '''


    return jsonify(response), status