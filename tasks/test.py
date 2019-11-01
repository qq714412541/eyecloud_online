# -*- coding: utf-8 -*-

from tasks.celery_config import app
import requests

@app.task
def fetch_url(url):
    resp = requests.get(url)
    print(resp.status_code)
    return resp.status_code

def test_celery(urls):
    for url in urls:
        fetch_url.delay(url)

def test_notify():
    request_body = {
        'id': 1,
        'results':[{
            'type': '正常',
            'probabelity': 0.99999,
            'direction': '右眼'
        }]
    }
    r = requests.post('http://localhost:5000/api/notify/fundusImage', json=request_body, headers={'Content-Type': 'application/json;charset=utf-8'})
    print(r.text)
    print(r)

# test_celery(['https://github.com', 'https://www.baidu.com'])