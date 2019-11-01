# -*- coding: utf-8 -*-

from services.encryption import EncrptionService
from services.base_code import BasecodeService
import requests, json
basecode = BasecodeService()

COMMON_HEADERS = {
    'Content-Type': 'application/json;charset=utf-8'
}

HOST_PORT = 'http://172.18.8.142:22222'
POST_EXAM_URL = HOST_PORT + '/api/commit/postExam'
FUNDUS_IMAGE_URL = HOST_PORT + '/api/commit/fundusImage'
SURFACE_IMAGE_URL = HOST_PORT + '/api/commit/surfaceImage'
APP_ID = '1'
APP_KEY = 'e10adc3949ba59abbe56e057f20f883e'

def test_post_exam():

    path = ['left.1.jpg', 'left.2.jpg']
    data = {
        'examId': 'ffc693501b334bc6a65d3f9b1c0d7aad110300034',
        'sickName': '永泽^黄',
        'sickSource': 'source',
        'sickSex':'man',
        'sickAge':'12',
        'noteStr': 'note',
        'image': basecode.enc(path)
    }

    sign = EncrptionService.sign_data(data, APP_KEY)
    print(sign)

    request = {
        'appId': APP_ID,
        'timeStr': '2019-09-08 18:02:46',
        'signStr': sign,
        'data': data
    }
    response = requests.post(
        url=POST_EXAM_URL,
        json=request,
        headers=COMMON_HEADERS
    )
    print(response.text)
    print(response)

def test_fundus_image():
    pass

def test_surface_image():
    pass


if __name__ == "__main__":
    test_post_exam()
