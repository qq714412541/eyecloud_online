# -*- coding: utf-8 -*-
import os
import json
from bunch import Bunch

from .util import mkdir_if_not_exist

def get_config_from_json(json_file):
    """
    change json file to dictionary
    """
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)  # load the dictionary

    config = Bunch(config_dict)  # convert dictionary into class

    return config, config_dict


def process_config(json_file):
    """
    read json and return configuration
    """
    config, _ = get_config_from_json(json_file)

    return config

