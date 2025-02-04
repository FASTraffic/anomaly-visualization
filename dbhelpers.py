import datetime as dt
import psycopg2
import psycopg2.extensions
from psycopg2 import Error as pg_Error
import pandas as pd
import time
import traceback
import pytz
import requests
import json
from zoneinfo import ZoneInfo

def load_config(file_path):
    """
    Load the configuration from a JSON file.

    :param file_path: Path to the JSON configuration file.
    :return: Configuration dictionary.
    """
    with open(file_path, 'r') as config_file:
        config = json.load(config_file)
    return config

def make_ts_db_connection():
    config = load_config('config.json')
    
    # Database connection details from config
    database_host = config['database_host']
    database_port = config['database_port']
    database_username = config['database_username']
    database_password = config['database_password']
    selected_database = config['selected_database']
    conn = psycopg2.connect(host=database_host,
                    port=database_port,
                    dbname=selected_database,
                    user=database_username,
                    password=database_password)

    return conn

def generate_db_transaction_ts_id():
    """
    Generates a transaction timestamp and ID value to store in the database. The database design is such that multiple
    rows may need to be written for a single logical group, in which case each row of the group should share this ID.
    :return: ID value
    """
    r3_tz = ZoneInfo('US/Central')
    return dt.datetime.now(r3_tz), int(time.time() * 1000)

def database_general_insert(query_text: str, query_values, existing_db_connection=None):
    """
    Common code to execute database insert, also populates values 'write_time' and 'db_update_id' shared by all tables,
        which should have named placeholders of the same name within query_text.
    A database connection is made inside this function and closed
    This function is generalized to work on a dictionary of values for a single query or a list of dictionaries for
        multiple queries of the same type. When multiple queries are made, they share the same 'write_time' and
        'db_update_id' and the entire set of queries is wrapped inside one database commit, for efficiency; this means
        that if one query fails, the entire transaction will be rolled back.

    :param query_text: INSERT query command with placeholders to be populated by execute() within DB connector
    :param query_values: dictionary of insert values with keys named in query_text, list of dicts if multiple queries
    :param existing_db_connection: (optional) existing database connection to use instead of initiating a new one
    :return: transaction-based ID value if successful, raises exception if unsuccessful
    """
    if existing_db_connection is None:
        # Attempt to connect and allow exception to kill function if it throws.
        # This utility function looks at the config file for the name of database to connect.
        ts_db_conn = make_ts_db_connection()
    else:
        ts_db_conn = existing_db_connection
    db_timestamp, db_update_id = generate_db_transaction_ts_id()

    if isinstance(query_values, list):
        query_dicts = query_values
    elif isinstance(query_values, dict):
        # Put in a list, so it can be iterated across like the multiple event data case.
        query_dicts = [query_values]
    else:
        raise TypeError("Got unexpected type for event_data. Should be dict (single event) or list of dict (multiple).")

    cur = None
    try:
        # Create a database cursor to use for the transactions.
        cur = ts_db_conn.cursor()
        for qdict in query_dicts:
            # Add transaction values that were created above, with correct placeholders from query stub.
            # Input `request_message_values` should have these placeholders also.
            # Populate query format string with data values using psycopg2 within execute().
            cur.execute(query=query_text, vars={'write_time': db_timestamp, 'db_update_id': db_update_id, **qdict})
        ts_db_conn.commit()
        return db_update_id
    except Exception as e:
        # Attempt to roll back the transactions.
        # This might fail if the database connection is problematic, and it isn't caught earlier.
        ts_db_conn.rollback()
        # Re-raise the exception, since it is supposed handled where this function is called and the message queue
        #   isn't passed into the function to deal with the error.
        raise e
    finally:
        # Wrap cursor close inside a try/except and ignore the exception if it occurs.
        # This will ignore any exception that happens here and allow any prior exception to propagate.
        try:
            cur.close()
            # Only close connection if it was made in this function scope.
            if existing_db_connection is None:
                ts_db_conn.close()
        # Should get AttributeError if cursor couldn't be created from connection, psycopg2.Error if there's
        #   something wrong with the cursor, and it can't be closed.
        except (AttributeError, pg_Error):
            pass

def get_query(query_name):
    if query_name == "last_hour":
        return """SELECT * 
        FROM raw_messages.raw_links
        WHERE link_update_time >= NOW() - INTERVAL '1 hour' """
    elif query_name == "10_mins":
        return """SELECT * 
        FROM raw_messages.raw_links
        WHERE link_update_time >= NOW() - INTERVAL '10 minutes' """
    elif query_name == "insert":
        return """
        INSERT INTO evaluations.ft_aed (
            write_time,
            db_update_id,
            rds_update_time,
            milemarker,
            lane_id,
            direction,
            reconstruction_error_rgcn,
            reconstruction_error_gat,
            reconstruction_error_gcn,
            threshold_rgcn,
            threshold_gat,
            threshold_gcn
        ) VALUES (%(write_time)s, %(db_update_id)s, %(rds_update_time)s, %(milemarker)s, %(lane_id)s, %(direction)s, %(reconstruction_error_rgcn)s, 
                  %(reconstruction_error_gat)s, %(reconstruction_error_gcn)s, %(threshold_rgcn)s, %(threshold_gat)s, %(threshold_gcn)s)
        ON CONFLICT (rds_update_time, milemarker, lane_id) DO NOTHING
        """
    else:
        raise NotImplementedError("Query not yet implemented. Check spelling.")

def get_last_10_mins():
    conn = make_ts_db_connection()

    try:
        with conn.cursor() as cur:
            query = get_query('10_mins')
            cur.execute(query)
            data = cur.fetchall()
    finally:
        conn.close()

    return pd.DataFrame(data)

def insert_predictions(query_values):
    query_text = get_query("insert")

    db_id = database_general_insert(query_text=query_text, query_values=query_values)