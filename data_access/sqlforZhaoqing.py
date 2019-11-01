
# -*- coding: utf-8 -*-
import json

from data_access.mysql_Zhaoqing import MysqlDatabase

class AppZhaoqing():

    def __init__(self):
        self.mysql = MysqlDatabase()

    def beforesqlZhaoqing(self, app_id, app_key, zhaoqing_time, exam_data, status):
        rep = 0
        exam_id = str(exam_data['examId'])
        zhaoqing_time = str(zhaoqing_time)
        app_id = str(app_id)

        print(exam_data)

        res = self.mysql.execute_sql('SELECT * FROM zhaoqing WHERE examId=%s', (exam_id, ), False)
        print(res)
        if res == []:
            rep = 1
            self.mysql.execute_sql(('INSERT INTO zhaoqing(examId, appId, appKey, zhaoqingTime, sickName, sickSex, sickAge, \
                                    sickSource, noteStr, status) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'),
                                    (exam_data['examId'], app_id, app_key, zhaoqing_time, exam_data['sickName'],
                                    exam_data['sickSex'], exam_data['sickAge'], exam_data['sickSource'],
                                    exam_data['noteStr'], status))

            return rep

        else:
            rep = 0
            return rep



    def aftersqlZhaoqing(self, exam_id, ai_time, status, data_json):##需要加if限制数组或者str
        exam_id = str(exam_id)
        #reportid = str(reportid)
        ai_time  = str(ai_time)
        #fundusimage = json.dumps({original_path})
        #report = json.dumps({report})  ############?????????

        res = self.mysql.execute_sql(('UPDATE zhaoqing SET aiTime=%s, status=%s, dataJson=%s WHERE examId=%s'),
                                    (ai_time, status, data_json, exam_id))

        return res

    def checkforZhaoqing(self, exam_id):
        exam_id = str(exam_id)
        results = self.mysql.execute_sql('SELECT * FROM zhaoqing WHERE examId=%s', (exam_id, ), False)
        print(results)
        if results == []:
            response_body = {
                'code': 10004,
                'error': 'no task'
            }
        else:
            row = results[0]
            status = row[10]
            if status == 0:
                response_body = {  # pending
                    'code': 0,
                    'status': 0
                }
            else:
                data_json = row[11]
                data = {
                    'examId': exam_id,
                    'sickName': row[3],
                    'sickAge': row[4],
                    'sickSex': row[5],
                    'sickSource': row[6],
                    'noteStr': row[7],
                    'zhaoqingTime': row[8],
                    'aiTime': row[9],
                    'status': 1,
                    'reports': json.loads(data_json)
                }
                response_body ={
                    'code': 1,
                    'data': data
                }

            print(response_body)
        return response_body



