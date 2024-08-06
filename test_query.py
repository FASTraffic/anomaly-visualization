import psycopg as pg
import json
import pandas as pd
 
def load_config(file_path):
    """
    Load the configuration from a JSON file.

    :param file_path: Path to the JSON configuration file.
    :return: Configuration dictionary.
    """
    with open(file_path, 'r') as config_file:
        config = json.load(config_file)
    return config

def get_query(query_name):
    if query_name == "last_hour":
        return """SELECT * 
        FROM raw_messages.raw_links
        WHERE link_update_time >= NOW() - INTERVAL '1 hour' """
    else:
        raise NotImplementedError("Query not yet implemented. Check spelling.")

if __name__=="__main__":
    config = load_config('config.json')
    
    # Database connection details from config
    database_host = config['database_host']
    database_port = config['database_port']
    database_username = config['database_username']
    database_password = config['database_password']
    selected_database = config['selected_database']
    conn = pg.connect(host=database_host,
                    port=database_port,
                    dbname=selected_database,
                    user=database_username,
                    password=database_password)

    print(conn)
    with conn.cursor() as cur:
        query = get_query('last_hour')
        cur.execute(query)
        data = cur.fetchall()

        if data:
            df = pd.DataFrame(data)
            print(df.head())

    conn.close()