# -*- coding: utf-8 -*-
import os, json
from celery import Celery
from kombu import Exchange, Queue

CONFIG_PATH = './config/redis.json'
with open(CONFIG_PATH, 'r') as config_file:
    config = json.load(config_file)
    HOST = config['host']
    PORT = config['port']
    DATABASE = config['database']

REDIS_URL = 'redis://%s:%d/%d' % (HOST, PORT, DATABASE)
RESULT_BACKEND = REDIS_URL
BROKER = REDIS_URL

TASKS = [
    'tasks.test',
    'tasks.fundus_image',
    'tasks.surface_image',
]

class Config(object):
    broker_url = REDIS_URL
    result_backend = REDIS_URL

    task_serializer = 'msgpack'
    result_serializer = 'json'
    # result_expires = 60*60*24
    accept_content = ['json','msgpack']

    include = [
        'tasks.test',
        'tasks.fundus_image',
        'tasks.surface_image',
    ]

    task_queues = (
        Queue('celery'),
        Queue('default', routing_key='default'),
        Queue('test', routing_key='test.#'),
        Queue('fundus', routing_key='fundus.#'),
        Queue('surface', routing_key='surface.#')
    )
    task_routes = (
        {
            'tasks.test.fetch_url': {
                'queue': 'test',
                'routing_key': 'test.fetch'
            }
        },
        {
            'tasks.fundus_image.mobile_fundus_analysis': {
                'queue': 'fundus',
                'routing_key': 'fundus.mobile'
            }
        },
        {
            'tasks.fundus_image.online_fundus_analysis': {
                'queue': 'fundus',
                'routing_key': 'fundus.online'
            }
        },
        {
            'tasks.surface_image.surface_image': {
                'queue': 'surface',
                'routing_key': 'surface.surface'
            }
        }
    )


app = Celery()
app.config_from_object(Config())

