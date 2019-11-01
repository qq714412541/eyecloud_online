# -*- coding: utf-8 -*-

class CheckService():

    @staticmethod
    def check_json_key(data, key_dict, optional_key_dict=None):
        check_result = True
        for key, key_type in key_dict.items():
            check_result = check_result and (key in data) and (CheckService.check_type(data[key], key_type))
        if optional_key_dict != None:
            for key, key_type in optional_key_dict.items():
                check_result = check_result and (key in data) and (CheckService.check_type(data[key], key_type))
        return check_result

    @staticmethod
    def check_int(value):
        return isinstance(value, int)

    @staticmethod
    def check_str(value):
        return isinstance(value, str)

    @staticmethod
    def check_dict(value):
        return isinstance(value, dict)

    @staticmethod
    def check_list(value):
        return isinstance(value, list)

    @staticmethod
    def check_type(value, data_type):
        if data_type == 'int':
            return CheckService.check_int(value)
        elif data_type == 'str':
            return CheckService.check_str(value)
        elif data_type == 'dict':
            return CheckService.check_dict(value)
        elif data_type == 'list':
            return CheckService.check_list(value)
        else:
            raise ValueError('No such type: %s' % (data_type))
