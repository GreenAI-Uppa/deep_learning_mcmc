import time

import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px
import streamlit as st

# Notes : 
#       Header = paragraphe de présentation + diagramme
#       5 box résumés de la latency

# selecteur de batch pour visualiser l'évo de la loss 
# (devrait ne pas converger en théorie hormis pour j4 car données diff au cours des 200 it)
#       graph d'évolution de la loss 
#       graphe d'évolution de l'AR

st.set_page_config(
     page_title="Controle du tuyau MCMC",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",
 )

st.title('MCMC monitoring')

# ==============================================================================
# définition du squelette de la page

with st.container(): # Présentation & diagramme
    _, main_col, _ = st.columns([1,3,1])

# ajouter un bouton de démarrage / arret du streaming 
names = ['j4', 'p8', 'p4', 'p2', 'j2']
def get_data(device_name):
    '''getting logs data from sshfs conexion'''
    return pd.read_csv(f'{device_name}/data', sep=';')

def get_latency(device_name):
    '''getting latency data from sshfs conexion'''
    return pd.read_csv(f'{device_name}/latency', sep=';')

def manage_data(dfs: dict):
    '''Merging logs dataframe & formatting to plot'''
    df = pd.DataFrame()
    for k, v in dfs.items():
        v['device'] = k
        df = df.append(v)
        
    df.dense = df.dense.apply(pd.to_numeric, errors='coerce')
    df.conv = df.conv.apply(pd.to_numeric, errors='coerce')
    
    bardf = df.melt(["id_batch", "device", "x", "loss"])
    return bardf[["device", "variable", "value"]].groupby(["device", "variable"]).mean().reset_index(), df

placeholder = st.empty()

while True:
        
    with placeholder.container(): # Latency box
        col_bj4, col_bp8, col_bp4, col_bp2, col_bj2 = st.columns([1,1,1,1,1])
        col_loss, col_ar = st.columns([5,5])
        
    # ==============================================================================
    dfs = {name: get_data(device_name=name) for name in names}
    df, df_loss = manage_data(dfs=dfs)    
    
    latencies = {name: get_latency(device_name=name) for name in names}
    
    # ==============================================================================
    # latency 
    # ==============================================================================
    # col_bj4, col_bp8, col_bp4, col_bp2, col_bj2 -> print last latency ?
    l = [[latencies.get(name)['lecture'].tolist()[-1], latencies.get(name)['envoie'].tolist()[-1]] for name in names]
        
    l = [[l[a][0], l[a+1][1]] for a in range(len(l)) if a < (len(l)-1)] + [[l[len(l)-1][0], l[0][1]]]

    for col, name, lat in zip([col_bj4, col_bp8, col_bp4, col_bp2, col_bj2], names, l):
        with col as c:
            # select last one and print
            st.metric(label=f"{name} latency (in s)", value=round(lat[0],2), delta=round(lat[1],2), delta_color="off")

    # ==============================================================================
    # AR & Loss
    # ==============================================================================
    fig = px.bar(df, x="device", y="value", 
                color="variable", barmode="group")
    col_ar.write(fig)
    
    fig = px.line(df_loss, x='x', y='loss', color='device', symbol="device")
    # loss par batch c'est mieux !!!!
    col_loss.write(fig)
    
    
    time.sleep(2)
    
    