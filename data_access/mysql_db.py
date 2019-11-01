# -*- coding: utf-8 -*-
import os, json
from mysql.connector import Error
from mysql.connector.connection import MySQLConnection
from mysql.connector.pooling import MySQLConnectionPool

class MysqlDatabase:

    def __init__(self, host, port, database, user, password):
        self.pool = MySQLConnectionPool(
            pool_name="connection_pool", pool_size=32,
            pool_reset_session=True, host=host, port=port,
            database=database, user=user, password=password
        )

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
