import pandas as pd 
import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import plotly.figure_factory as ff
from streamlitmetrics import metric, metric_row



def app():
    
    #titre de la page 
    st.title("Dashboard business")

    #bloc de sélection des poids par l'utilisateur
    st.markdown("<h2 style='text-align: center;'>Définition des coûts & bénéfices</h2>", unsafe_allow_html=True)

    c5, c6 = st.columns(2)
    c7, c8 = st.columns(2)

    TP = c5.number_input("Gain d'un client sans défaut détecté",value = 1,min_value = -1000, max_value = 1000)
    FN = c6.number_input("Perte d'un client sans défaut non-détecté",value = -1,min_value = -1000, max_value = 1000)
    TN = c7.number_input("Gain d'un client faillible détecté",value = 1,min_value = -1000, max_value = 1000)
    FP = c8.number_input("Perte d'un client faillible non-détecté",value= -1,min_value = -1000, max_value = 1000)
    
    TP
    FN
    TN
    FP

    #chargement du CSV
    df = pd.read_csv("result_CBC_deploy.csv")
    
    #back-end de sélection du seuil de risque (càd probabilité d'être en classe 0 ou 1)
    range_of_thresh = np.arange(0.1,0.9,0.01)
    list_of_benef = []
    best_benef = -999999999999
    for i in range_of_thresh :
        predict_tmp = df["proba"].apply(lambda x:1 if x>= i else 0)
        cf_tmp = confusion_matrix(df["ground"],predict_tmp)
        cf_tmp[0][0] = cf_tmp[0][0]*TN
        cf_tmp[0][1] = cf_tmp[0][1]*FP
        cf_tmp[1][0] = cf_tmp[1][0]*FN
        cf_tmp[1][1] = cf_tmp[1][1]*TP
        benef_tmp = cf_tmp.sum()
        if benef_tmp > best_benef : 
            best_benef = benef_tmp
            best_thresh = np.round(i,2).item()
        list_of_benef.append(benef_tmp)
    df_benef = pd.DataFrame()
    df_benef["Seuil"] = np.around(range_of_thresh,2)
    df_benef["Bénéfice"]= list_of_benef
    
    benef_point = []
    thresh_point = []
    benef_point.append(best_benef)
    thresh_point.append(best_thresh)
    
    #front-end de sélection du seuil de risque (càd probabilité d'être en classe 0 ou 1)
    st.markdown("<h2 style='text-align: center;'>-----------------------------------------------------------</h2>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Seuil de risque</h2>", unsafe_allow_html=True)
    
    Thresh = st.slider("Choix du seuil",min_value = 0.1,max_value = 0.9,value = best_thresh)
    
    
    #back-end des metrics selon les paramètres choisis par l'utilisateur
    df['predicted_new'] = df["proba"].apply(lambda x:1 if x>= Thresh else 0)
    
    fig = px.line(df_benef, x="Seuil", y="Bénéfice", title = "Evolution du bénéfice en fonction du seuil de risque")
    fig.update_traces(line=dict(color = "blue"))
    fig.add_trace(go.Scatter(
        mode='markers',
        name="Meilleur seuil estimé",
        x=thresh_point,
        y=benef_point,
        marker=dict(
            color="green",
            size=15,
            )))
    fig.add_vline(Thresh)
    
    cf= confusion_matrix(df["ground"],df['predicted_new'])
    cf_weighted = cf.copy()
    cf_weighted[0][0] = cf_weighted[0][0]*TN
    cf_weighted[0][1] = cf_weighted[0][1]*FP
    cf_weighted[1][0] = cf_weighted[1][0]*FN
    cf_weighted[1][1] = cf_weighted[1][1]*TP
    
    benef = cf_weighted.sum()
    
    x = ['Client sans défaut connu', 'Client faillible connu']
    y = ['Client sans défaut prédit', 'Client faillible prédit']
    
    str_list =  [[str(cf_weighted[0,0])+" $", str(cf_weighted[0,1])+" $"],
                [str(cf_weighted[1,0])+" $", str(cf_weighted[1,1])+" $"]]
    

    fig_mat_priced = ff.create_annotated_heatmap(cf_weighted,x=x,y=y,annotation_text=str_list, colorscale="BuPu")
    fig_mat_priced.layout.update({'title': 'Evaluation impact business'})
    fig_mat = ff.create_annotated_heatmap(cf,x=x,y=y, colorscale="BuGn")
    fig_mat.layout.update({'title': 'Matrice de confusion'})

    #front-end des metrics selon les paramètres choisis par l'utilisateur
    st.plotly_chart(fig, use_container_width = True)

    st.markdown("<h2 style='text-align: center;'>-----------------------------------------------------------</h2>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Impact de vos choix</h2>", unsafe_allow_html=True)

    c9, c10 = st.columns(2)

    c9.plotly_chart(fig_mat, use_container_width=True)
    c10.plotly_chart(fig_mat_priced, use_container_width=True)
    
    metric_row({"Meilleur bénéfice estimé": best_benef,
                "Risque estimé": best_thresh,
                "Bénéfice avec votre choix de seuil": df_benef[df_benef['Seuil']==Thresh]["Bénéfice"].values[0],
                "Risque choisi": Thresh})
