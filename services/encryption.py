# -*- coding: utf-8 -*-
import json, hashlib

class EncrptionService():

    @staticmethod
    def sign_data(data, app_key):
        data_str = json.dumps(data) + app_key
        return hashlib.md5(data_str.encode('utf8')).hexdigest()