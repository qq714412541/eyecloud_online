# -*- coding: utf-8 -*-

import pymysql
import json

class TaskStatusService():

    def __init__(self):
        pass

    def insert_task(self):
        pass

    def update_status(self, task_id):
        pass


class pyc:

  def pytomy(self, r,g,response_body):

      con = pymysql.connect(host='10.20.71.67', port=33333, user='root', db='mysql', passwd='qq714142541')
      mycur = con.cursor()
      #r = "DEMO-00004"
      r=str(r)
      g=str(g)
      sql = "SELECT * FROM jsontest \
                   WHERE reportid=" + '"' + r + '"'+ "and genre="+  '"' + g + '"'
      print(sql)
      mycur.execute(sql)
      # 获取所有记录列表
      results = mycur.fetchall()
      print(results)

      if results == ():
          response_body = {
              'code': 10002,
              'error': 'no task'
          }
      else:

          for row in results:
              datasql = row[1]
              status = row[3]
              print(datasql)

              # 打印结果
              if status == 0:
                  response_body = {  # pending
                      'code': 0,
                      'status': 0
                  }
              else:
                  datasql1=json.loads(datasql)
                  response_body = {  # finished
                      'code': 0,
                      'status': 1,
                      'result': datasql1
                  }

              print(response_body)

      return response_body
      con.close()
