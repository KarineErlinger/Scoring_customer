import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from streamlitmetrics import metric, metric_row
import re

def app():
    #titre de la page
    st.title("Dashboard Scoring Customer")  

    #box d'input du numéro de client
    client_ID = st.text_input("Numéro du client")

    #regex pour récupérer uniquement les chiffres dans le texte entré
    client_ID = ''.join(re.findall('[0-9]+', client_ID))

    df_CBC = pd.read_csv("result_CBC_deploy.csv")
    
    #if qui déclenche la fonction "display" lorsqu'un numéro de client existant est entré
    #ou affiche un message d'erreur si le numéro de client est mauvais
    if client_ID:
        if client_ID in df_CBC["SK_ID_CURR"].values.astype(str):
            display(client_ID)
        else:
            st.markdown("<h1 style='text-align: center; color: red;'>Merci de vérifier votre numéro de client</h1>", unsafe_allow_html=True)


def display (client_ID):
    
    #import du dataframe
    df_CBC = pd.read_csv("result_CBC_deploy.csv")

    #on passe les ID clients en string pour matcher avec l'input box
    df_CBC["SK_ID_CURR"] = df_CBC["SK_ID_CURR"].astype(str)

    #récupération de la ligne du client dans le dataframe
    client_output = df_CBC[df_CBC["SK_ID_CURR"] == client_ID]

    #bloc qui permet d'afficher les metrics avec leurs définitions
    st.markdown("<h2 style='text-align: center;'>Prédiction du profil du client - payeur</h2>", unsafe_allow_html=True)
    metrics = metric_row(
        {
            "Numéro de client": client_output["SK_ID_CURR"].values[0],
            "Classe (0 = sans défaut, 1 = faillible)": client_output["predict"].values[0],
            "% de risques que le client soit faillible": str(np.round(client_output["proba"].values[0]*100, 2)) + "%"
        }
    )
    metrics

    st.markdown("<h2 style='text-align: center;'>-----------------------------------------------------------</h2>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Profil du client & comparaison</h2>", unsafe_allow_html=True)

    #création d'un dataframe vide qui servira à contenir les lignes de la même classe que le client (0 ou 1)
    classe_output = pd.DataFrame()

    #remplissage de ce dataframe
    if client_output["predict"].values == 0:
        classe_output = df_CBC[df_CBC["predict"] == 0].mean()
    elif client_output["predict"].values == 1:
        classe_output = df_CBC[df_CBC["predict"] == 1].mean()
    else:
        print("error")

    #Bloc : création d'un dataframe contenant les données les plus importantes (features importance) pour classifier le client
    df_all = pd.DataFrame()

    df_all["Population"] = ["Client", "Classe", "Ensemble"]

    df_all["AMT_REQ_CREDIT_BUREAU_YEAR"] = [client_output["AMT_REQ_CREDIT_BUREAU_YEAR"].values[0],
    classe_output["AMT_REQ_CREDIT_BUREAU_YEAR"].mean(), df_CBC["AMT_REQ_CREDIT_BUREAU_YEAR"].mean()]

    df_all["HOUR_APPR_PROCESS_START"] = [client_output["HOUR_APPR_PROCESS_START"].values[0],
    classe_output["HOUR_APPR_PROCESS_START"].mean(), df_CBC["HOUR_APPR_PROCESS_START"].mean()]

    df_all["CNT_CHILDREN"] = [client_output["CNT_CHILDREN"].values[0],
    classe_output["CNT_CHILDREN"].mean(), df_CBC["CNT_CHILDREN"].mean()]

    df_all["OBS_30_CNT_SOCIAL_CIRCLE"] = [client_output["OBS_30_CNT_SOCIAL_CIRCLE"].values[0],
    classe_output["OBS_30_CNT_SOCIAL_CIRCLE"].mean(), df_CBC["OBS_30_CNT_SOCIAL_CIRCLE"].mean()]

    #création de colonnes pour afficher les plots côte à côte (ici par paire)
    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)

    #barplots des données du client seul, de son groupe (classe 0 ou 1) et de l'ensemble des clients
    #Client : donnée seule. Groupe & ensemble : Moyenne de la colonne.
    plot1 = c1.plotly_chart(px.bar(df_all, x='Population', y='AMT_REQ_CREDIT_BUREAU_YEAR', title="Nombre de demandes au bureau de crédit sur le client, par jour & an", color='Population'), use_container_width=True)
    plot1
    plot2 = c2.plotly_chart(px.bar(df_all, x='Population', y='HOUR_APPR_PROCESS_START', title="Vers quelle heure le client a-t-il fait sa demande ?", color='Population'), use_container_width=True)
    plot2
    plot3 = c3.plotly_chart(px.bar(df_all, x='Population', y='CNT_CHILDREN', title="Nombre d'enfants du client", color='Population'), use_container_width=True)
    plot3
    plot4 = c4.plotly_chart(px.bar(df_all, x='Population', y='OBS_30_CNT_SOCIAL_CIRCLE', title="Aptitude de l'entourage du client à l'aider financièrement", color='Population'), use_container_width=True)
    plot4

    #plot des feature importance extraits de la modélisation
    st.markdown("<h2 style='text-align: center;'>-----------------------------------------------------------</h2>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Pourquoi ces éléments ont été sélectionnés ?</h2>", unsafe_allow_html=True)
    feature_imp = pd.read_csv('feature_imp_top10_renamed.csv')
    st.plotly_chart(px.bar(feature_imp, x="Variable", y="Pourcentage d'importance", title='Importance des variables dans la classification', color="Pourcentage d'importance"), use_container_width=True)

    return()




