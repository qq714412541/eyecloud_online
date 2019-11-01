# -*- coding: utf-8 -*-

from flask import Flask, request, redirect, jsonify, Blueprint
from services.logger import LogService
import json

notify_api = Blueprint('notify', __name__)

logger = LogService()

@notify_api.route('/surfaceImage', methods=['POST'])
def notify_surfaceImage():
    response = {}
    data = request.get_json()
    logger.info(data)
    
    id_ = data['id']
    level = data['level']
    cornealArea = data['cornealArea']
    damagedArea = data['damagedArea']

    response = {
        'code': 0
    }

    return jsonify(response)

@notify_api.route('/fundusImage', methods=['POST'])
def notify_fundusImage():
    response = {}
    data = request.get_json()
    logger.info(data)

    id_ = data['id']
    results = data['results']

    response = {
        'code': 0
    }

    return jsonify(response)

@notify_api.route('/test', methods=['GET'])
def test():
    return 'Hello, notify!'