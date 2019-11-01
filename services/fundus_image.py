# -*- coding: utf-8 -*-

import os, time, base64
from werkzeug.utils import secure_filename
import tasks.fundus_image as fundus

from services.base_code import BasecodeService
basecode = BasecodeService()

class FundusImageService():

    def __init__(self):
        pass

    def run_online_fundus(self, id_, images_path, callback):
        f_status_list = fundus.online_fundus_analysis.delay(id_, images_path, callback)
        return f_status_list

    def run_mobile_fundus(self, app_id, time, sign, exam_data):

        original_path = basecode.dec(exam_data['image'])
        timestr = fundus.mobile_fundus_analysis.delay(app_id, time, sign, original_path, exam_data)

        return timestr