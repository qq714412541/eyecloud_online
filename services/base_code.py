import base64,cv2,time,os,json


#CONFIG_PATH = './config/path.json'
#with open(CONFIG_PATH, 'r') as config_file:
    #config = json.load(config_file)
    #PICTURE_PATH = config['picturePath']

PICTURE_PATH = "/home/pic"   #different server has different place
class BasecodeService:

    def enc(self,path):

        img = list(range(len(path)))

        for i in range(len(path)):

            img[i] = base64.b64encode(cv2.imencode('.jpg', cv2.imread(path[i]))[1]).decode()
        return img

    def dec(self,img):

        path = list(range(len(img)))

        for i in range(len(img)):
            hms = '%s.jpg' % (time.time())

            path[i] = os.path.join(PICTURE_PATH, hms)

            img_byte = base64.b64decode(img[i])
            file = open(path[i], 'wb')

            file.write(img_byte)
            file.close()
        return path