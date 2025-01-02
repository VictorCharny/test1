import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
 
# Charger un fichier GeoJSON ou un CSV prétraité
@st.cache_data
def load_data():
    # Remplacer par l'URL ou un fichier CSV contenant les informations géospatiales
    url = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements.geojson"
    return pd.read_json(url)
 
# Charger les données
gdf = load_data()
 
# Extraire les départements
departements = gdf["features"].apply(lambda x: x["properties"]["nom"])
 
# Carte Folium
m = folium.Map(location=[46.603354, 1.888334], zoom_start=6)
for feature in gdf["features"]:
    folium.GeoJson(
        feature["geometry"],
        tooltip=feature["properties"]["nom"],
    ).add_to(m)
 
# Afficher la carte dans Streamlit
st.title("Carte interactive des départements français")
st_folium(m, width=700, height=500)






