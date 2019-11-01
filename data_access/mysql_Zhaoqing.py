# -*- coding: utf-8 -*-
import os, json
from mysql.connector import Error
from mysql.connector.connection import MySQLConnection
from mysql.connector.pooling import MySQLConnectionPool

CONFIG_PATH = './config/mysql.json'
with open(CONFIG_PATH, 'r') as config_file:
    config = json.load(config_file)
    HOST = config['zhaoqing']['host']
    PORT = config['zhaoqing']['port']
    USER = config['zhaoqing']['user']
    PASSWORD = config['zhaoqing']['password']
    DATABASE = config['zhaoqing']['database']


'''with open(CONFIG_PATH, 'r') as config_file:
    config = json.load(config_file)
    HOST = config['eyecloud']['host']
    PORT = config['eyecloud']['port']
    USER = config['eyecloud']['user']
    PASSWORD = config['eyecloud']['password']
    DATABASE = config['eyecloud']['database']'''

class MysqlDatabase:

    @staticmethod
    def get_connection_pool():
        MysqlDatabase.pool = MySQLConnectionPool(pool_name="connection_pool",
                                                pool_size=32,
                                                pool_reset_session=True,
                                                host=HOST,
                                                port=PORT,
                                                database=DATABASE,
                                                user=USER,
                                                password=PASSWORD)

    def __init__(self):
        if not hasattr(MysqlDatabase, 'pool'):
           MysqlDatabase.get_connection_pool()

    def execute_sql(self, sql_statement, parameters, commit=True):
        record = []
        try:
            # Get connection object from a pool
            conn = MysqlDatabase.pool.get_connection()
            if conn.is_connected():
                cursor = conn.cursor()
                cursor.execute(sql_statement, parameters)
                if commit:
                    conn.commit()
                else:
                    record = cursor.fetchall()
        except Error as err :
            raise err
        finally:
            # Close database connection.
            if conn.is_connected():
                cursor.close()
                conn.close()
            return record
