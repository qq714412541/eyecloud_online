# -*- coding: utf-8 -*-

from data_access.app import AppDao

class AppService():

    def __init__(self):
        self.app_dao = AppDao()

    def add_app(self, app_id, app_key):
        return self.app_dao.add_app(app_id, app_key)

    def get_app_key(self, app_id):
        result = self.app_dao.get_app_key(app_id)
        if len(result) != 0: # no data
            result = result[0][0]
        return result

    def delete_app(self, app_id):
        return self.app_dao.delete_app(app_id)
