# coding:utf-8

from flask import Flask, request, redirect, url_for
from flask import jsonify, send_from_directory, send_file, safe_join, abort
from werkzeug.utils import secure_filename
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
import tornado.options
import os
import time
import json
import uuid
import datetime

STORE_DIR = '/root/pic'
ALLOWED_EXTENSIONS = set(['pdf', 'png', 'jpg', 'jpeg'])
IP_NODE = 2886862059

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/report')
def hello():
    return 'hello 尼古拉斯赵四'

@app.route('/report/<path:filepath>', methods=['GET'])
def dowload_report(filepath):
    filename = filepath
    safe_path = safe_join(STORE_DIR, filename)

    try:
        return send_file(safe_path, as_attachment=True)
    except FileNotFoundError:
        abort(404)


@app.route('/upload', methods=['POST'])
def upload():
    response, status = {}, 200
    print(request.form)

    f = request.files['file']

    if f and allowed_file(f.filename):

        hms = time.time()
        current_datetime = datetime.datetime.fromtimestamp(hms)
        time_dir = os.path.join(str(current_datetime.year), str(current_datetime.month),
                                str(current_datetime.day))
        directory = os.path.join(STORE_DIR, time_dir)

        if not os.path.exists(directory):
            os.makedirs(directory)

        end = f.filename.split('.')[-1]
        upload_filename = '%s.%s' % (uuid.uuid1(node=IP_NODE), end)
        upload_path = os.path.join(directory, secure_filename(upload_filename))

        f.save(upload_path)

        response = {
            'code': 0,
            'reportPath': os.path.join(time_dir, upload_filename)
        }
    else:
        response = {
            'code': 10001
        }

    return jsonify(response), status

def app_debug():
    app.debug = True
    app.run(host='0.0.0.0', port=5000)

def main():
    tornado.options.define("port", default=5000, help="run on the given port", type=int)
    tornado.options.parse_command_line()
    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(tornado.options.options.port, address="0.0.0.0")
    IOLoop.instance().start()

if __name__ == "__main__":
    main()
