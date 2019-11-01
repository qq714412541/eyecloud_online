from data_access.sqlforZhaoqing import AppZhaoqing
pyc =  AppZhaoqing()
'''response_body={}
res = pyc.checkforEyecloud(414,1,response_body)
print('#########')
print(res)'''

rep = int
rep = pyc.beforesqlEyecloud(414,0,1,rep)

print(rep)

'''data = {
    'adas': '啊实打实',
    'd':'da'
}
pyc.aftersqlEyecloud(data,414,1,1)'''


'''response_body={}
res = pyc.checkforZhaoqing(3,'打算',response_body)
print('#########')
print(res)'''


'''a = str
a = pyc.checktest(414,a)
print(a)'''


'''data = {
        'examId': 3,
        'sickName': '打算',
        'sickSource': 'source',
        'sickSex':'man',
        'noteStr': 'note',
        'sickAge': 18

    }
rep = int
rep = pyc.beforesqlZhaoqing('asdadasz111','asda2111','sada131',data,'/pic/1.jpg',0,rep)

print(rep)'''

#res = pyc.aftersqlZhaoqing(3,['/pic/1111.jpg','/pic/2222.jpg'],['12a1231','ads231'],'124as',['41212','1241'],['1','0'],'v1.21',1,['aaaa','bbbb'])
#res = pyc.aftersqlZhaoqing(3,['/pic/2222.jpg'],['ads231'],'124as',['1241'],['0'],'v1.21',1,['bbbb'])

#print(res)




#INSERT INTO zhaoqing(examid,appid,appkey,time1,sickname,sicksex,sickage,sicksource,note,fundusimage) VALUES(1,'asdadasz111','asda2111','sada131','打算','man',18,'source','note','/pic/1.jpg')
#update zhaoqing set reportpath='/ p i c / 3 1 3 1 . j p g',reportid=%s,time2=%s,aitime=%s,abnormal=%s,version=%s,status=%s where examid=%s',
# ('/ p i c / 3 1 3 1 . j p g', '1 2 a s d   a s d a s', '124as', '4 1 2 1 2', '1', 'v1.21', 1, '3')

#('update zhaoqing set reportpath='/pic/3131.jpg#/pic/12.jpg',reportid='12asd asdas#adsada 12',time2='124as',aitime='41212#1241',abnormal='1#0',version='v1.21',status= 1 where examid='3'',
 #('/pic/3131.jpg#/pic/12.jpg', '12asd asdas#adsada 12', '124as', '41212#1241', '1#0', 'v1.21', 1, '3'))