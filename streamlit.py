import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
 
# Charger les données GeoJSON
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements.geojson"
    return gpd.read_file(url)
 
gdf = load_data()
 
# Créer une carte interactive avec Folium
m = folium.Map(location=[46.603354, 1.888334], zoom_start=6)  # Centre de la France
for _, row in gdf.iterrows():
    folium.GeoJson(
        row['geometry'],
        tooltip=row['nom']
    ).add_to(m)
 
# Afficher la carte dans Streamlit
st.title("Carte interactive des départements français")
st_folium(m, width=700, height=500)













