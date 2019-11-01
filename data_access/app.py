# -*- coding: utf-8 -*-

from data_access.mysql_db import MysqlDatabase

class AppDao():

    def __init__(self):
        self.mysql = MysqlDatabase()

    def add_app(self, app_id, app_key):
        return self.mysql.execute_sql(('INSERT INTO appTable(appId, appKey) VALUES(%s, %s)'), (app_id, app_key))

    def get_app_key(self, app_id):
        return self.mysql.execute_sql('SELECT appKey FROM appTable WHERE appId = %s', (app_id, ), False)

    def delete_app(self, app_id):
        return self.mysql.execute_sql('DELETE FROM appTable WHERE appId = %s', (app_id, ))