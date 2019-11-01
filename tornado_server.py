# -*- coding: utf-8 -*-

from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from routes.main import app
import tornado.options
import os

def main():

    LOGS_FILE_PATH = "./logs"
    if not os.path.exists(LOGS_FILE_PATH):
        os.makedirs(LOGS_FILE_PATH)

    tornado.options.define("port", default=5000, help="run on the given port", type=int)
    tornado.options.parse_command_line()
    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(tornado.options.options.port,address="0.0.0.0")
    IOLoop.instance().start()

if __name__ == "__main__":
    main()

## python tornado_server.py --log_file_prefix=./logs/5000.log --log_rotate_mode=time --log_rotate_when=H --log_rotate_interval=1
