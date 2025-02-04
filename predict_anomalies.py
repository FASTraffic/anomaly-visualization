import psycopg as pg
import json
import pandas as pd
import ast
import datetime
import numpy as np
import torch

from torch_geometric.data import Data
from datetime import timedelta
from zoneinfo import ZoneInfo

from parameters import GraphAEParameters, GATAEParameters, RSTAEParameters
from models import GraphAE, GATAE, RelationalSTAE 
from training import load_model
from datautils import load_best_parameters, generate_edges, generate_relational_edges, process_as_df, normalize_data
from datautils import milemarkers as all_milemarkers
from dbhelpers import make_ts_db_connection, insert_predictions, get_query, get_last_10_mins
import time

def load_gcn():
    optimal_hyperparams = load_best_parameters('gcn_v2')
    hyperparams = GraphAEParameters(
        num_features=3,
        latent_dim=optimal_hyperparams['latent_dim'],
        gcn_hidden_dim=optimal_hyperparams['gcn_hidden_dim'],
        dropout=optimal_hyperparams['dropout'],
        num_gcn=optimal_hyperparams['num_gcn']
    )

    ae = load_model(GraphAE, hyperparams, 'gcn')

    return ae

def gcn_input_process(data):
    data = data.apply(lambda x: x.fillna(x.mean()), axis=0)
    normalized_data = normalize_data(data)
    time_pred = normalized_data.iloc[-1]['Time']
    time_pred = time_pred - pd.Timedelta(minutes=1) # 1 minute before so data has time to come in

    static_edges = generate_edges(milemarkers=list(range(49)))

    gcn_input = normalized_data[normalized_data['Time'] == time_pred].to_numpy(dtype=np.float32)[:,3:6]
    curr_graph = Data(x=torch.tensor(gcn_input, dtype=torch.float32), edge_index=static_edges)

    return curr_graph, time_pred

def load_stg_gat():
    optimal_hyperparams = load_best_parameters('gat')

    hyperparams = GATAEParameters(
        num_features=3,
        latent_dim=optimal_hyperparams['latent_dim'],
        gcn_hidden_dim=optimal_hyperparams['gcn_hidden_dim'],
        dropout=optimal_hyperparams['dropout'],
        num_layers=optimal_hyperparams['num_layers'],
        num_heads=optimal_hyperparams['num_heads']
    )

    ae = load_model(GATAE, hyperparams, 'gat')

    return ae, optimal_hyperparams['timesteps']

def sequential_input_process(data, timestep, thirty_seconds_delay):
    data = data.apply(lambda x: x.fillna(x.mean()), axis=0)
    normalized_data = normalize_data(data)
    time_preds = []
    for i in range(timestep-1, -1, -1):
        time_preds.append(np.unique(normalized_data['Time'])[-1-thirty_seconds_delay-i])


    relational_edges, relations = generate_relational_edges(milemarkers=list(range(49)), timesteps=timestep)
    inputs = []
    for time_pred in time_preds:
        gat_input = normalized_data[normalized_data['Time'] == time_pred].to_numpy(dtype=np.float32)[:,3:6]
        inputs.append(gat_input)
    
    inputs = np.concatenate(inputs)

    pyg_data = Data(x=torch.tensor(inputs, dtype=torch.float32), edge_index=relational_edges, edge_attr=torch.tensor(relations, dtype=torch.long))

    return pyg_data

def load_rgcn_ae():
    optimal_hyperparams = load_best_parameters('rstae')

    hyperparams = RSTAEParameters(
        num_features=3,
        latent_dim=optimal_hyperparams['latent_dim'],
        gcn_hidden_dim=optimal_hyperparams['gcn_hidden_dim'],
        dropout=optimal_hyperparams['dropout'],
        num_gcn=optimal_hyperparams['num_gcn']
    )

    ae = load_model(RelationalSTAE, hyperparams, 'rstae')

    return ae, optimal_hyperparams['timesteps']

def load_thresholds():
    gcn_thresh = np.load('saved_models/gcn_thresh.npy')
    gat_thresh = np.load('saved_models/gat_thresh.npy')
    rstae_thresh = np.load('saved_models/rstae_thresh.npy')

    return gcn_thresh, gat_thresh, rstae_thresh

def compute_error(predictions, data):
    error = (predictions - data)**2
    error = np.mean(error, axis=1)
    return error

def format_predictions(time, gcn_error, gcn_thresh, gat_error, gat_thresh, rstae_error, rstae_thresh):
    result = []
    milemarker_list = all_milemarkers.repeat(4)
    lane_list = np.tile(np.arange(1, 5), 49)

    for mile, lane, gcn_e, gcn_t, gat_e, gat_t, rstae_e, rstae_t in zip(milemarker_list, lane_list, gcn_error, gcn_thresh, gat_error, gat_thresh, rstae_error, rstae_thresh):
        r3_tz = ZoneInfo('US/Central')
        localized_time = time.astimezone(r3_tz)

        rds_update_time = localized_time
        result.append({'rds_update_time': rds_update_time, 
                       'milemarker': float(mile), 
                       'lane_id': int(lane),
                       'direction': "Westbound", 
                       'reconstruction_error_gcn': float(gcn_e), 
                       'reconstruction_error_gat': float(gat_e),
                       'reconstruction_error_rgcn': float(rstae_e), 
                       'threshold_gcn': float(gcn_t),
                       'threshold_gat': float(gat_t),
                       'threshold_rgcn': float(rstae_t)})

    return result

if __name__=="__main__":
    # How many 30 seconds the predictions should be delayed by
    # Right now, this is 1 minute
    thirty_seconds_delay = 2

    start_time = time.time()

    # Query the database for the past 10 minutes of RDS data
    df = get_last_10_mins()
    # Preprocess that data to remove duplicate times, unnecessary data, etc.
    processed_data = process_as_df(df)

    # Loading GCN model weights and input preprocessing. Applying GCN model to data.
    gcn_ae = load_gcn()
    gcn_input, time_pred = gcn_input_process(processed_data)
    gcn_recons = gcn_ae(gcn_input).detach().numpy()

    # GAT model
    gat_ae, gat_timestep = load_stg_gat()
    gat_data = sequential_input_process(processed_data, gat_timestep, thirty_seconds_delay)
    gat_recons = gat_ae(gat_data).detach().numpy()

    # RSTAE model
    rstae, rstae_timestep = load_rgcn_ae()
    rstae_data = sequential_input_process(processed_data, rstae_timestep, thirty_seconds_delay)
    rstae_recons = rstae(rstae_data).detach().numpy()

    # Loading thresholds for anomaly detection
    gcn_thresh, gat_thresh, rstae_thresh = load_thresholds()

    # Fill missing data to be used for reconstruction error computation
    processed_data =  processed_data.apply(lambda x: x.fillna(x.mean()), axis=0)
    current_data = normalize_data(processed_data[processed_data['Time']==time_pred]).to_numpy()[:,3:6]

    # Compute reconstruction errors
    gcn_error = compute_error(gcn_recons, current_data)
    gat_error = compute_error(gat_recons, current_data)
    rstae_error = compute_error(rstae_recons, current_data)
    
    # Format predictions into query format
    predictions = format_predictions(time_pred, gcn_error, gcn_thresh, gat_error, gat_thresh, rstae_error, rstae_thresh)

    # Write predictions to DB
    insert_predictions(predictions)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f'Wrote predictions to DB at {datetime.datetime.now()}. Took {elapsed_time:.2f} seconds.')
