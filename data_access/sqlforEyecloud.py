
# -*- coding: utf-8 -*-
import json

from data_access.mysql_Eyecloud import MysqlDatabase

class AppEyecloud():

    def __init__(self):
        self.mysql = MysqlDatabase()

    #def insert_initEyecloud(self,reportid,status,genre):
    #    return self.mysql.execute_sql(('INSERT INTO jsontest(reportid,status,genre) VALUES(%s, %s,%s)'), (reportid,status,genre))


    #def insert_initZhaoqing(self,appid,appkey, time1,  exam_data,fundusimage):
    #    return self.mysql.execute_sql(('INSERT INTO test(examid,appid,appkey,time1,sickname,sicksex,sickage,sicksource,note,fundusimage) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'),
    #                                  (exam_data['examId'],appid,appkey,time1,exam_data['sickName'],exam_data['sickSex'],exam_data['sickAge'],exam_data['sickSource'],exam_data['noteStr'],fundusimage))

    #def add_app(self, app_id, app_key):
    #    return self.mysql.execute_sql(('INSERT INTO appTable(appId, appKey) VALUES(%s, %s)'), (app_id, app_key))

    def beforesqlEyecloud(self, reportid,status,genre,rep):
        rep=int
        genre = str(genre)
        reportid = str(reportid)
        res = self.mysql.execute_sql('SELECT * FROM jsontest WHERE reportid = %s and genre=%s', (reportid,genre ), False)
        print(res)
        if res == []:
            rep = 1
            self.mysql.execute_sql(('INSERT INTO jsontest(reportid,status,genre) VALUES(%s, %s,%s)'),
                                   (reportid, status, genre))

            return rep

        else:
            rep = 0


            return rep

    def aftersqlEyecloud(self,data,reportid,status,genre):
        reportid = str(reportid)
        data_json = json.dumps(data)
        return self.mysql.execute_sql('update jsontest set data=%s,status=%s where reportid=%s and genre=%s',(data_json,status,reportid,genre))






    def checkforEyecloud(self,reportid,genre,response_body):
        reportid = str(reportid)
        genre = str(genre)
        results = self.mysql.execute_sql('SELECT * FROM jsontest WHERE reportid = %s and genre = %s ',(reportid,genre),False)
        print(results)
        if results == []:
            response_body = {
                'code': 10002,
                'error': 'no task'
            }
        else:

            for row in results:
                datasql = row[1]
                status = row[3]
                print(datasql)
                print(type(datasql))

                # 打印结果
                if status == 0:
                    response_body = {  # pending
                        'code': 0,
                        'status': 0
                    }
                else:
                    datasql1 = json.loads(datasql)
                    response_body = {  # finished
                        'code': 0,
                        'status': 1,
                        'result': datasql1
                    }

                print(response_body)

        return response_body


####select 需要两个 where来确认




