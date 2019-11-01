
# -*- coding: utf-8 -*-

from flask import Flask
from routes.notify import notify_api
from routes.commit import commit_api
from routes.status import status_api

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

@app.route('/', methods=['GET'])
def hello():
    return 'Hello World!'

app.register_blueprint(notify_api, url_prefix = '/api/notify')
app.register_blueprint(commit_api, url_prefix='/api/commit')
app.register_blueprint(status_api, url_prefix='/api/status')

if __name__ == '__main__':
    app.run()