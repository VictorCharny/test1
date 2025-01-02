# -*- coding: utf-8 -*-
"""streamlit.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1U9SdneS60j_sQ_KNVaQFF9aShV9nmCA4
"""

import pandas as pd


import streamlit as st
import pandas as pd
import altair as alt

pip install geopandas
pip install follium 

st.set_page_config(
    page_title="Athéna",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")
st.title("Athéna")

data1=pd.read_csv('data1.csv')
data2=pd.read_csv('data2.csv')
data3=pd.read_csv("data3.csv")

df=data2
df['Jour'] = pd.to_datetime(df['timestamp'], errors='coerce', dayfirst=True).dt.day
df['Mois'] = pd.to_datetime(df['timestamp'], errors='coerce', dayfirst=True).dt.month
df['Année'] = pd.to_datetime(df['timestamp'], errors='coerce', dayfirst=True).dt.year
df['Heure'] = pd.to_datetime(df['timestamp'], errors='coerce', dayfirst=True).dt.strftime('%H:%M:%S')

df["zone"]=0
for i in range(len(df)):
  if (df.loc[i, 'User_id1'].endswith('I') or
      df.loc[i, 'User_id2'].endswith('I')):
    df.loc[i, "zone"]="italie"
  elif (df.loc[i, 'User_id1'].endswith('G') or
        df.loc[i, 'User_id2'].endswith('A')):
    df.loc[i, "zone"]="allemagne"
  else:
    df.loc[i, "zone"]="espagne"




with st.sidebar:
    st.title('Messages dans Athena')
    
    month_list = list(df.Mois.unique())[::-1]
    
    selected_month = st.selectbox('Choisissez un mois', month_list, index=len(month_list)-1)
    df_selected_month = df[df.Mois == selected_month]
    

    color_theme_list = ['blues', 'cividis', 'greens', 'inferno', 'magma', 'plasma', 'reds', 'rainbow', 'turbo', 'viridis']
    selected_color_theme = st.selectbox('Select a color theme', color_theme_list)

import folium
import geopandas as gpd
import pandas as pd

def map(data):
    # Calcul des occurrences pour chaque zone
    valeur1 = (data["zone"] == "italie").sum()
    valeur2 = (data["zone"] == "allemagne").sum()
    valeur3 = (data["zone"] == "espagne").sum()

    # Lecture du fichier GeoJSON des départements
    deppartement = gpd.read_file("contour-des-departements.geojson", encoding='utf-8')
    
    # Sélection des départements pour chaque région
    dep1 = deppartement[deppartement["nom"] == "Pyrénées-Atlantiques"]
    dep2 = deppartement[deppartement["nom"] == "Pyrénées-Orientales"]
    dep3 = deppartement[deppartement["nom"] == "Ariège"]
    dep4 = deppartement[deppartement["nom"] == "Hautes-Pyrénées"]
    dep5 = deppartement[deppartement["nom"] == "Haute-Garonne"]
    region_front1 = pd.concat([dep1, dep2, dep3, dep4, dep5])

    dep1 = deppartement[deppartement["nom"] == "Alpes-Maritimes"]
    dep2 = deppartement[deppartement["nom"] == "Alpes-de-Haute-Provence"]
    dep3 = deppartement[deppartement["nom"] == "Hautes-Alpes"]
    dep4 = deppartement[deppartement["nom"] == "Savoie"]
    dep5 = deppartement[deppartement["nom"] == "Haute-Savoie"]
    region_front2 = pd.concat([dep1, dep2, dep3, dep4, dep5])

    dep1 = deppartement[deppartement["nom"] == "Haut-Rhin"]
    dep2 = deppartement[deppartement["nom"] == "Bas-Rhin"]
    dep3 = deppartement[deppartement["nom"] == "Moselle"]
    region_front3 = pd.concat([dep1, dep2, dep3])

    # For region 1
    region_font1 = region_front1.copy()
    region_font1["valeur"] = valeur3  # Valeur uniforme pour cette région

    # For region 2
    region_font2 = region_front2.copy()
    region_font2["valeur"] = valeur1

    # For region 3
    region_font3 = region_front3.copy()
    region_font3["valeur"] = valeur2

    # Concatenation de toutes les régions
    region = pd.concat([region_font1, region_font2, region_font3])

    # Création de la carte
    m = folium.Map(location=[44.0, 2.0], zoom_start=6, control_scale=True)

    # Ajouter un fond de carte
    folium.TileLayer('cartodb positron').add_to(m)

    # Création de la carte choroplèthe
    chloropleth = folium.Choropleth(
        geo_data=region,  # dataframe géospatial
        data=region,
        columns=['nom', 'valeur'],  # Colonnes avec les valeurs à afficher
        key_on='feature.properties.nom',  # Correspondance avec la géométrie
        fill_opacity=0.5,  # Opacité de la couleur
        line_opacity=1,  # Opacité des frontières
        legend_name='Répartition des messages'  # Légende
    )

    # Ajout de la couche choroplèthe à la carte
    chloropleth.add_to(m)

    # Ajout des tooltips
    chloropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(
            fields=['nom', 'valeur'],  # Champs à afficher dans le tooltip
            aliases=['Département', 'Valeur'],  # Alias pour chaque champ
            labels=True,  # Affichage des labels
            sticky=True  # Garde le tooltip affiché
        )
    )

    # Retourner la carte
    return m






col = st.columns((1.5, 4.5, 2), gap='medium')

























