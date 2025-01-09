
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import geopandas as gpd
import plotly.express as px
import numpy as np
import seaborn as sns

# Premier graphique : Occurrences des messages par jour
import plotly.graph_objects as go
import pandas as pd



import streamlit as st
import pandas as pd
import altair as alt
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from streamlit_extras.metric_cards import style_metric_cards




st.set_page_config(
    page_title="Athéna",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded")


css='''
[data-testid="metric-container"] {
    width: fit-content;
    margin: auto;
}

[data-testid="metric-container"] > div {
    width: fit-content;
    margin: auto;
}

[data-testid="metric-container"] label {
    width: fit-content;
    margin: auto;
}
'''

st.markdown(f'<style>{css}</style>',unsafe_allow_html=True)

alt.themes.enable("dark")




data1=pd.read_csv('data1.csv')
data2=pd.read_csv('data2.csv')
data3=pd.read_csv("data3.csv")
data4=pd.read_csv("data4.csv")
data4=data4.rename(columns={'Message_ID': 'Message_id'})


#df=data2
df=pd.merge(data2,data3,on="Message_id")
dfchat=pd.merge(data2,data4,on=["User_id1","User_id2","Message_id"])



df['Jour'] = pd.to_datetime(df['timestamp'], errors='coerce', dayfirst=True).dt.day
df['Mois'] = pd.to_datetime(df['timestamp'], errors='coerce', dayfirst=True).dt.month
df['Année'] = pd.to_datetime(df['timestamp'], errors='coerce', dayfirst=True).dt.year
df['Heure'] = pd.to_datetime(df['timestamp'], errors='coerce', dayfirst=True).dt.strftime('%H')



dfchat['Jour'] = pd.to_datetime(dfchat['timestamp'], errors='coerce', dayfirst=True).dt.day
dfchat['Mois'] = pd.to_datetime(dfchat['timestamp'], errors='coerce', dayfirst=True).dt.month
dfchat['Année'] = pd.to_datetime(dfchat['timestamp'], errors='coerce', dayfirst=True).dt.year
dfchat["Heure"] = pd.to_datetime(dfchat['timestamp'], errors='coerce', dayfirst=True).dt.strftime('%H')




dfchat["zone"]=0
for i in range(len(dfchat)):
  if (dfchat.loc[i, 'User_id1'].endswith('I') or
      dfchat.loc[i, 'User_id2'].endswith('I')):
    dfchat.loc[i, "zone"]="italie"
  elif (dfchat.loc[i, 'User_id1'].endswith('G') or
        dfchat.loc[i, 'User_id2'].endswith('G')):
    dfchat.loc[i, "zone"]="allemagne"
  else:
    dfchat.loc[i, "zone"]="espagne"




df["zone"]=0
df["langue"]=0
for i in range(len(df)):
  if (df.loc[i, 'User_id1'].endswith('I') or
      df.loc[i, 'User_id2'].endswith('I')):
    df.loc[i, "zone"]="italie"
  elif (df.loc[i, 'User_id1'].endswith('G') or
        df.loc[i, 'User_id2'].endswith('G')):
    df.loc[i, "zone"]="allemagne"
  else:
    df.loc[i, "zone"]="espagne"


df["langue"]=0
for i in range(len(df)):
  if (df.loc[i, 'User_id1'].endswith('I')):
      
    df.loc[i, "langue"]="Italien"
  elif (df.loc[i, 'User_id1'].endswith('G')):
    df.loc[i, "langue"]="Allemand"
  
  elif (df.loc[i, 'User_id1'].endswith('E')):
    df.loc[i, "langue"]="Espagnol"
  else:
    df.loc[i, "langue"]="Français"

dfchat=pd.merge(dfchat,data3,on="Message_id")


with st.sidebar:
    st.title('Dashboard Athena')
    
    
    months = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"]

# Créer un dictionnaire pour associer les mois aux indices numériques (1-12)
    month_dict = {month: i+1 for i, month in enumerate(months)}

# Créer une liste des mois présents dans les données, triée
    month_list = list(df.Mois.unique())[::-1]
    month_list = sorted(month_list)
    month_list = [month for month in month_list if not np.isnan(month)]
    month_list = [int(x) for x in month_list]

# Création de la liste des mois en français à partir des indices dans `month_list`
    month_names = [months[i-1] for i in month_list]

# Sélection du mois avec Streamlit
    selected_month = st.selectbox('Choisissez un mois', month_names, index=len(month_list)-1)

# Convertir le mois sélectionné en numéro via le dictionnaire `month_dict`
    month_number = month_dict[selected_month]
    selected_month1=selected_month
    selected_month=month_number

# Filtrer les données pour le mois sélectionné
    df_selected_month = df[df.Mois == month_number]
    dfchat_selected_month=dfchat[dfchat.Mois==month_number]

    st.markdown("---")



    zone_list=list(df.zone.unique())[::-1]
    zone_list.append("All")
    selected_zone=st.selectbox('Choisissez une zone frontalière',zone_list,index=len(zone_list)-1)
    
    if selected_zone=="All":
        df_selected_zone=df
        dfchat_selected_zone=dfchat
    else:
        df_selected_zone=df[df.zone== selected_zone]
        dfchat_selected_zone=dfchat[dfchat.zone==selected_zone]
    st.markdown("---")
    st.sidebar.image("SNCF_Réseau.png")


    









def map(data):
    # Calcul des occurrences pour chaque zone
    valeur1 = (data["zone"] == "italie").sum()
    valeur2 = (data["zone"] == "allemagne").sum()
    valeur3 = (data["zone"] == "espagne").sum()

    # Lecture du fichier GeoJSON des départements
    departement = gpd.read_file("contour-des-departements.geojson", encoding='utf-8')

    # Sélection des départements pour chaque région frontalière
    dep1 = departement[departement["nom"] == "Pyrénées-Atlantiques"]
    dep2 = departement[departement["nom"] == "Pyrénées-Orientales"]
    dep3 = departement[departement["nom"] == "Ariège"]
    dep4 = departement[departement["nom"] == "Hautes-Pyrénées"]
    dep5 = departement[departement["nom"] == "Haute-Garonne"]
    region_front1 = pd.concat([dep1, dep2, dep3, dep4, dep5])

    dep1 = departement[departement["nom"] == "Alpes-Maritimes"]
    dep2 = departement[departement["nom"] == "Alpes-de-Haute-Provence"]
    dep3 = departement[departement["nom"] == "Hautes-Alpes"]
    dep4 = departement[departement["nom"] == "Savoie"]
    dep5 = departement[departement["nom"] == "Haute-Savoie"]
    region_front2 = pd.concat([dep1, dep2, dep3, dep4, dep5])

    dep1 = departement[departement["nom"] == "Haut-Rhin"]
    dep2 = departement[departement["nom"] == "Bas-Rhin"]
    dep3 = departement[departement["nom"] == "Moselle"]
    region_front3 = pd.concat([dep1, dep2, dep3])

    # Pour les régions frontalières, affecter des valeurs spécifiques
    region_front1["valeur"] = valeur3
    region_front2["valeur"] = valeur1
    region_front3["valeur"] = valeur2

    # Créer une copie de l'ensemble des départements pour inclure toutes les régions
    region_complete = departement.copy()

    # Initialiser toutes les valeurs à 0 (ou toute autre valeur par défaut)
    region_complete["valeur"] = 0

    # Intégrer les valeurs des régions frontalières dans l'ensemble complet
    region_complete.loc[region_front1.index, "valeur"] = valeur3
    region_complete.loc[region_front2.index, "valeur"] = valeur1
    region_complete.loc[region_front3.index, "valeur"] = valeur2

    # Préparer les données pour la carte Plotly
    geojson = region_complete.__geo_interface__  # Convertir GeoDataFrame en GeoJSON

    # Créer la carte choroplèthe avec Plotly
    fig = px.choropleth(
        region_complete,
        geojson=geojson,
        locations=region_complete.index,  # Utilisation des indices des départements comme identifiant
        color="valeur",
        hover_name="nom",
        hover_data=["valeur"],
        color_continuous_scale="blues",  # Choisir un schéma de couleurs
        labels={"valeur": "Messages"},
        
    )

    # Mettre à jour la projection pour la carte de la France
    fig.update_geos(fitbounds="locations", visible=False)

    fig.update_layout(
    template='plotly_dark',
    plot_bgcolor='rgba(0, 0, 0, 0)',
    paper_bgcolor='rgba(0, 0, 0, 0)',
    margin=dict(l=0, r=0, t=0, b=0),
    height=350
    )



    return fig
def calcul(df):  #nombre de messages par régions
   zone_list=list(df.zone.unique())[::-1]

   values1=[]
   a=len(df)
   for i in(zone_list):
      b = len(df[df["zone"] == i])
      
      values1.append(b)
   return(values1)

def calculate(df,year): #différence avec vle mois précédent
   values=[]
   selected_year=df[df['Mois'] == year].reset_index()
   previous_year_data = df[df['Mois'] == year - 1].reset_index()
   selected_month = selected_year['zone'].unique()

   for i in (selected_month):
      a = previous_year_data[previous_year_data['zone'] == i]
      b = selected_year[selected_year['zone'] == i]
      c=len(b)-len(a)
      values.append(c)
   return(values)


def generate_metric_cards_graphically(df, year):
    # Appel de la fonction calculate pour obtenir les différences
    values = calculate(df, year)
    selected_year = df[df['Mois'] == year].reset_index()
    selected_months = selected_year['zone'].unique()
    
    # Créer une figure pour afficher les cartes
    fig, axes = plt.subplots(len(selected_months), 1, figsize=(5, 3 * len(selected_months)))
    
    if len(selected_months) == 1:
        axes = [axes]  # Pour le cas où il n'y a qu'une seule zone
    
    # Pour chaque zone, afficher une "metric card"
    for ax, i, value in zip(axes, selected_months, values):
        # Couleur conditionnelle: vert pour positif, rouge pour négatif
        color = 'lightgreen' if value > 0 else 'lightcoral'
        
        # Affichage du texte avec la couleur conditionnelle
        ax.text(0.5, 0.5, f"Zone: {i}\nDifférence avec le mois précédent: {value}", fontsize=14, ha='center', va='center', color='black')
        
        # Appliquer la couleur de fond
        ax.set_facecolor(color)
        
        # Enlever les ticks x et y pour une carte propre
        ax.set_xticks([])  
        ax.set_yticks([])  
        
        # Titre de la carte
        ax.set_title(f'Carte métrique pour la zone {i}')
    
    # Ajuster l'espacement pour ne pas chevaucher les éléments
    plt.tight_layout()
    
    # Afficher le graphique dans Streamlit
    st.pyplot(fig)
def matrice(df):
   a=df[df["zone"]=="espagne"]
   b=df[df["zone"]=="italie"]
   c=df[df["zone"]=="allemagne"]
   return(a,b,c)

col1,col2,col3,col4,col5=st.columns([1,1,1,1,1],gap="small")
with col1:
    
    st.metric(label="Nombre total d'utilisateurs",value=len(data1),delta=len(data1))
    style_metric_cards(border_left_color="#003366", background_color="#222222")

with col2:
    
    st.metric(label="Nombre  d'utilisateurs italiens",value=len(data1[data1["Langue"]=="Italien"]),delta=len(data1[data1["Langue"]=="Italien"]))
with col3:
    
    st.metric(label="Nombre  d'utilisateurs allemand",value=len(data1[data1["Langue"]=="Allemand"]),delta=len(data1[data1["Langue"]=="Allemand"]))
with col4:
    
    st.metric(label="Nombre  d'utilisateurs Espagnol",value=len(data1[data1["Langue"]=="Espagnol"]),delta=len(data1[data1["Langue"]=="Espagnol"]))
with col5:
    st.metric(label="Nombre  d'utilisateurs Français",value=len(data1[data1["Langue"]=="Français"]),delta=len(data1[data1["Langue"]=="Français"]))

    
st.markdown("---")

                





# Créer trois colonnes avec Streamlit
col1,  col3 = st.columns([1, 1])  # La première et la troisième colonne sont petites, la deuxième est grande


# Code à placer dans la colonne 1
with col1:
    st.markdown('#### Nombre de messages sur un an par zone ')

    # Créer une liste vide pour stocker les résultats
    result = []



    # Itérer sur tous les mois uniques dans les données
    for i in month_list:
        # Filtrer les données pour chaque mois
        df_month = df[df["Mois"] == i]
        
        # Débogage pour vérifier si les données existent pour chaque mois
        
        #st.write(df_month.head())  # Afficher un aperçu des données pour chaque mois
        
        if not df_month.empty:
            # Séparer les données par zone (Espagne, Italie, Allemagne)
            a, b, c = matrice(df_month)
            
            # Calculer la longueur des messages dans chaque zone
            a_length = len(a)  # Longueur des messages en Espagne
            b_length = len(b)  # Longueur des messages en Italie
            c_length = len(c)  # Longueur des messages en Allemagne
            total=a_length+b_length+c_length
            
            # Ajouter ces résultats dans la liste sous forme de ligne [mois, espagne_length, italie_length, allemagne_length]
            result.append([i, a_length, b_length, c_length,total])
        else:
            st.write(f"Aucune donnée pour le mois {i}.")

    # Convertir la liste en DataFrame
    if result:
        df_result = pd.DataFrame(result, columns=["Mois", "Espagne", "Italie", "Allemagne","Total"])

        # Afficher le DataFrame dans Streamlit
        #st.write("DataFrame des résultats:", df_result)  # Affichage du DataFrame
        #st.dataframe(df_result)  # Affichage interactif

        # Créer les courbes pour chaque zone
        fig = go.Figure()
        months = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"]

        fig.add_trace(go.Scatter(x=months, y=df_result["Espagne"], mode='lines+markers', name="Espagne", line=dict(color='royalblue')))
        fig.add_trace(go.Scatter(x=months, y=df_result["Italie"], mode='lines+markers', name="Italie", line=dict(color='dodgerblue')))
        fig.add_trace(go.Scatter(x=months, y=df_result["Allemagne"], mode='lines+markers', name="Allemagne", line=dict(color='skyblue')))
        fig.add_trace(go.Scatter(x=months, y=df_result["Total"], mode='lines+markers', name="Total", line=dict(dash='dash', color='mediumblue')))

        # Personnalisation du graphique
        fig.update_layout(
            title="Nombre de messages par Zone et par Mois",
            xaxis_title="Mois",
            yaxis_title="Nombre de messages",
            showlegend=True
        )

        # Afficher le graphique avec Streamlit
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.write("Aucun message n'a été trouvé pour les zones et mois spécifiés.")



with col3:
    

    # Créer une liste vide pour stocker les résultats
    result = []

    # Vérification que month_list contient bien des mois
    #st.write("Month List:", month_list)

    # Itérer sur tous les mois uniques dans les données
    for i in month_list:
        # Filtrer les données pour chaque mois
        df_month = df[df["Mois"] == i]
        
        # Débogage pour vérifier si les données existent pour chaque mois
        #st.write(f"Messages pour le mois {i}:")
        #st.write(df_month.head())  # Afficher un aperçu des données pour chaque mois
        
        if not df_month.empty:
            # Séparer les données par zone (Espagne, Italie, Allemagne)
            a, b, c = matrice(df_month)
            
            # Compter le nombre de messages dans chaque zone
            a_count = len(a)
            b_count = len(b)
            c_count = len(c)
            
            # Ajouter ces résultats dans la liste sous forme de ligne [zone, espagne, italie, allemagne]
            result.append([i, a_count, b_count, c_count])
        else:
            st.write(f"Aucune donnée pour le mois {i}.")

    # Convertir la liste en DataFrame
    if result:
        df_result = pd.DataFrame(result, columns=["Mois", "Espagne", "Italie", "Allemagne"])

       

        # Transformer en format long (melt)
        df_long = df_result.melt(id_vars=["Mois"], value_vars=["Espagne", "Italie", "Allemagne"], 
                                 var_name="Zone", value_name="Messages")
        


        mois_dict = {
    1: 'Janvier', 2: 'Février', 3: 'Mars', 4: 'Avril', 5: 'Mai', 6: 'Juin',
    7: 'Juillet', 8: 'Août', 9: 'Septembre', 10: 'Octobre', 11: 'Novembre', 12: 'Décembre'
}

# Appliquer le remplacement des mois par les noms
        df_long["Mois"] = df_long["Mois"].replace(mois_dict)

# Définir l'ordre chronologique des mois
        mois_order = ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin', 
              'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre']

# Convertir la colonne "Mois" en catégorie avec l'ordre spécifié
        df_long["Mois"] = pd.Categorical(df_long["Mois"], categories=mois_order, ordered=True)


# Créer la heatmap avec Altair
        heatmap = alt.Chart(df_long).mark_rect().encode(
            x=alt.X('Mois:N', title="Mois",sort=mois_order), 
            y=alt.Y('Zone:N', title="Zone"), 
            color=alt.Color('Messages:Q', scale=alt.Scale(scheme='blues'), title="Nombre de messages"),
            tooltip=['Mois', 'Zone', 'Messages']
            ).properties(
            width=600,
            height=400,
            title="Heatmap des messages pour Athena"
)


        st.altair_chart(heatmap, use_container_width=True)

    else:
        st.write("Aucun message n'a été trouvé pour les zones et mois spécifiés.")
   
st.markdown("---")
col2, col1, col3 = st.columns([0.3, 1, 0.5])  # La première et la troisième colonne sont petites, la deuxième est grande

with col1:
    st.markdown(f"### Carte interactive  ")


    choropleth = map(df_selected_month)
    from streamlit.components.v1 import html
    html(choropleth._repr_html_(), height=600)  # Affichage de la carte avec une hauteur de 600px

    
    
with col2:
    st.markdown('#### Augmentations/Pertes')

    values1=calcul(df_selected_month)
    values=calculate(df,selected_month)
    label1="Espagne"
    label2="Allemagne"
    label3="Italie"
    st.metric(label=label1,value=values1[0],delta=values[0])
    st.metric(label=label2,value=values1[1],delta=values[1])
    st.metric(label=label3,value=values1[2],delta=values[2])

    
with col3:
   st.markdown('#### Répartition des messages ')
   result = calcul(df_selected_month)  # Calcul des résultats
   zone_list = ['Espagne', 'Allemagne', 'Italie']  # Remplacez par votre liste de zones réelles

   # Créer un graphique circulaire avec Plotly
   fig = px.pie(
       names=zone_list,  # Noms des zones
       values=result,  # Valeurs des zones
       title="Proportions des zones",  # Titre du graphique
       hole=0.5,  # Créer un "donut chart" (si vous préférez un cercle plein, définissez hole=0)
       color=zone_list,  # Ajouter des couleurs distinctes pour chaque zone
       color_discrete_sequence=px.colors.sequential.Cividis
  # Palette de couleurs
   )

   # Afficher le graphique avec Streamlit
   st.plotly_chart(fig)

st.markdown("---")




col1, col3 = st.columns([1, 1])  


with col1:

    
    
        #time_list =[#"Semaine","Jour","Heure"]
    time_list=["Jour","Heure"]



    selected_time = st.selectbox('Vision spécifique intervalle temporel', time_list, index=len(time_list)-1)


    if selected_time=="Jour":
        if selected_zone=="All":
            df4=df[df["Mois"]==selected_month]
            day_counts = df4['Jour'].value_counts().sort_index()
            fig = px.bar(
            x=day_counts.index,  # Jour
            y=day_counts.values,  # Occurrences
            labels={'x': 'Jour', 'y': 'Occurrences'},  # Labels pour les axes
            title="Messages/jour",  # Titre du graphique
            color=day_counts.values,  # Couleur basée sur la valeur
            color_continuous_scale='Blues',  # Palette de couleurs
   )     
            fig.update_layout(
            xaxis=dict(
                tickmode='linear',  # Afficher tous les ticks
                tickvals=day_counts.index,  # Les ticks correspondent aux valeurs d'heure
                tickangle=45,  # Incliner les labels pour une meilleure lisibilité
                title="Heure"
            ),
            yaxis=dict(
                title="Occurrences"
            )
        )
            st.plotly_chart(fig)


        else:

        
                df3 = df_selected_month[df_selected_month["zone"] == selected_zone]
                day_counts = df3['Jour'].value_counts().sort_index()
                fig = px.bar(
                x=day_counts.index,  # Jour
                y=day_counts.values,  # Occurrences
                labels={'x': 'Jour', 'y': 'Occurrences'},  # Labels pour les axes
                title="Messages/jour",  # Titre du graphique
                color=day_counts.values,  # Couleur basée sur la valeur
                color_continuous_scale='Blues',  # Palette de couleurs
   )        
                fig.update_layout(
                xaxis=dict(
                tickmode='linear',  # Afficher tous les ticks
                tickvals=day_counts.index,  # Les ticks correspondent aux valeurs d'heure
                tickangle=45,  # Incliner les labels pour une meilleure lisibilité
                title="Heure"
                ),
                yaxis=dict(
                title="Occurrences"
            )
        )   
                st.plotly_chart(fig)
                
             
            

   
   # Afficher le graphique avec Streamlit
        
    
    if selected_time=="Heure":
        if selected_zone=="All":
            df4=df[df["Mois"]==selected_month]
            day_counts = df4['Heure'].value_counts().sort_index()
            fig = px.bar(
            x=day_counts.index,  # Jour
            y=day_counts.values,  # Occurrences
            labels={'x': 'Heure', 'y': 'Occurrences'},  # Labels pour les axes
            title="Messages/heure",  # Titre du graphique
            color=day_counts.values,  # Couleur basée sur la valeur
            color_continuous_scale='Blues',  # Palette de couleurs
            )
            fig.update_layout(
            xaxis=dict(
                tickmode='linear',  # Afficher tous les ticks
                tickvals=day_counts.index,  # Les ticks correspondent aux valeurs d'heure
                tickangle=45,  # Incliner les labels pour une meilleure lisibilité
                title="Heure"
            ),
            yaxis=dict(
                title="Occurrences"
            )
        )
   
            st.plotly_chart(fig)
        else:
            df3 = df_selected_month[df_selected_month["zone"] == selected_zone]
            day_counts = df3['Heure'].value_counts().sort_index()
            fig = px.bar(
            x=day_counts.index,  # Jour
            y=day_counts.values,  # Occurrences
            labels={'x': 'Heure', 'y': 'Occurrences'},  # Labels pour les axes
            title="Messages/heure",  # Titre du graphique
            color=day_counts.values,  # Couleur basée sur la valeur
            color_continuous_scale='Blues',  # Palette de couleurs
   )    
            
            fig.update_layout(
            xaxis=dict(
                tickmode='linear',  # Afficher tous les ticks
                tickvals=day_counts.index,  # Les ticks correspondent aux valeurs d'heure
                tickangle=45,  # Incliner les labels pour une meilleure lisibilité
                title="Heure"
            ),
            yaxis=dict(
                title="Occurrences"
            )
        )
   
   # Afficher le graphique avec Streamlit
            st.plotly_chart(fig)


with col3:
    if selected_time=="Heure" or selected_time=="Jour" or selected_time=="Jour":
        st.markdown(f"### DataFrame pour le mois {selected_month1} et la zone {selected_zone} trié par {selected_time}")




    if selected_zone=="All":
        filtered_df = df[df["Mois"]==selected_month]
        filtered_df=filtered_df.sort_values(by=selected_time)
        if not filtered_df.empty:
            st.dataframe(filtered_df)
        else:
            st.write("Aucune donnée disponible pour cette sélection.")


    else:

        filtered_df = df[(df["Mois"] == selected_month) & (df["zone"] == selected_zone)]
        filtered_df=filtered_df.sort_values(by=selected_time)


        if not filtered_df.empty:
            st.dataframe(filtered_df)
        else:
            st.write("Aucune donnée disponible pour cette sélection.")   




st.markdown("---")
if selected_zone == "All":
    # Traitement des données pour toutes les zones
    depeche_counts = df_selected_month["depeche"].value_counts().reset_index()
    depeche_counts.columns = ['depeche', 'Occurrences']
    depeche_counts_sorted = depeche_counts.sort_values(by="Occurrences", ascending=False)

    depeche_counts_chat = dfchat_selected_month["depeche"].value_counts().reset_index()
    depeche_counts_chat.columns = ['depeche', 'Occurrences']

    depeche_counts_chat['Ratio'] = (depeche_counts_chat['Occurrences'] / len(dfchat_selected_month)) * 100
    depeche_counts['Ratio'] = (depeche_counts['Occurrences'] / len(df_selected_month)) * 100

    depeche_counts_sorted = depeche_counts.sort_values(by="Occurrences", ascending=False)
    depeche_counts_chat_sorted = depeche_counts_chat.sort_values(by="Occurrences", ascending=False)
else:
    # Filtrage des données en fonction de la zone sélectionnée
    df_selected_month = df_selected_month[df_selected_month["zone"] == selected_zone]
    depeche_counts = df_selected_month["depeche"].value_counts().reset_index()
    depeche_counts.columns = ['depeche', 'Occurrences']

    dfchat_selected_month = dfchat_selected_month[dfchat_selected_month["zone"] == selected_zone]
    depeche_counts_chat = dfchat_selected_month["depeche"].value_counts().reset_index()
    depeche_counts_chat.columns = ['depeche', 'Occurrences']

    # Calcul du ratio en pourcentage
    depeche_counts_chat['Ratio'] = (depeche_counts_chat['Occurrences'] / len(dfchat_selected_month)) * 100
    depeche_counts['Ratio'] = (depeche_counts['Occurrences'] / len(df_selected_month)) * 100

    # Tri des résultats
    depeche_counts_sorted = depeche_counts.sort_values(by="Occurrences", ascending=False)
    depeche_counts_chat_sorted = depeche_counts_chat.sort_values(by="Occurrences", ascending=False)

# Création de 3 colonnes avec Streamlit
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('#### Depeches les plus recommandées')
    st.dataframe(
        depeche_counts_sorted,
        column_order=("depeche", "Occurrences"),
        hide_index=True,
        width=None,
        column_config={
            "depeche": st.column_config.TextColumn("depeche"),
            "Occurrences": st.column_config.ProgressColumn(
                "Occurrences", format="%f", min_value=0, max_value=max(depeche_counts_sorted["Occurrences"])
            ),
        }
    )

st.markdown("----")
with col2:
    st.markdown('#### Ratio dépêches/chats')

    # Utilisation de Plotly pour générer le barplot avec le ratio en pourcentage
    fig = px.bar(
        depeche_counts_chat_sorted,
        x="depeche",
        y="Ratio",  # Utilisez 'Ratio' pour afficher les pourcentages
        labels={'depeche': 'Depeche', 'Ratio': 'Ratio (%)'},
        #title="Ratio des depeches par rapport au nombre total de chats"
    )

    # Affichage du graphique dans Streamlit
    st.plotly_chart(fig, use_container_width=True)

with col3:
    st.markdown('#### Ratio dépêche/messages')

    # Utilisation de Plotly pour générer le barplot avec le ratio en pourcentage
    fig = px.bar(
        depeche_counts_sorted,
        x="depeche",
        y="Ratio",  # Utilisez 'Ratio' pour afficher les pourcentages
        labels={'depeche': 'Depeche', 'Ratio': 'Ratio (%)'},
        #title="Ratio des depeches par rapport au nombre total de messages"
    )

    # Affichage du graphique dans Streamlit
    st.plotly_chart(fig, use_container_width=True)
col1, col2, col3 = st.columns(3)

with col1:
    if selected_zone=="All":
        #
        langues=["Français","Italien","Allemand","Espagnol","All"]
    elif selected_zone=="italie":
        langues=["Français","Italien","All"]
        #
    elif selected_zone=="allemagne":
        langues=["Français","Allemand","All"]
    else:
        #
        langues=["Français","Espagnol","All"]

    





    selected_langue=st.selectbox('Langue du message', langues, index=len(langues)-1)
    if selected_zone=="All":
        dfa=df_selected_month
        for i in (langues):
            a=0
            if i!="All":
                dfa2=df_selected_month[df_selected_month["langue"]==i]
                a=len(dfa2)
                st.metric(label=f" messages en {i}  zone frontalière {selected_zone} en {selected_month1}",value=a)
            else:
                dfa=df_selected_month
                st.metric(label=f" messages en {i}   zone frontalière {selected_zone} en {selected_month1}",value=len(dfa))
    else:
        dfa=df_selected_month[df_selected_month["zone"]==selected_zone]
        for i in (langues):
            a=0
            if i!="All":
                dfa1=dfa[dfa["langue"]==i]
                a=len(dfa1)
                st.metric(label=f" messages en {i}  zone frontalière {selected_zone} en {selected_month1}",value=a)
            else:
                dfa=dfa
                a=len(dfa)
                st.metric(label=f" messages en {i}  zone frontalière {selected_zone} en {selected_month1}",value=a)

with col2:
    st.markdown('#### Répartition des reccomandations')
    if selected_langue != "All":

        df=df_selected_month[df_selected_month["langue"]==selected_langue]
        a=df[df["Recom_num"]==1]
        b=df[df["Recom_num"]==2]
        c=df[df["Recom_num"]==3]
        a=len(a)
        b=len(b)
        c=len(c)
    # Créer un graphique circulaire avec Plotly
        fig = px.pie(
        names=[1,2,3],  # Noms des zones
        values=[a,b,c],  # Valeurs des zones
        #title="Répartition des reccomandations",  # Titre du graphique
        hole=0.5,  # Créer un "donut chart" (si vous préférez un cercle plein, définissez hole=0)
        color=[1,2,3],  # Ajouter des couleurs distinctes pour chaque zone
        color_discrete_sequence=px.colors.sequential.Cividis
  # Palette de couleurs
        )

   # Afficher le graphique avec Streamlit
        st.plotly_chart(fig)
    else:
        df=df_selected_month
        a=df[df["Recom_num"]==1]
        b=df[df["Recom_num"]==2]
        c=df[df["Recom_num"]==3]
        a=len(a)
        b=len(b)
        c=len(c)
    # Créer un graphique circulaire avec Plotly
        fig = px.pie(
        names=[1,2,3],  # Noms des zones
        values=[a,b,c],  # Valeurs des zones
        #title="Répartition des reccomandations",  # Titre du graphique
        hole=0.5,  # Créer un "donut chart" (si vous préférez un cercle plein, définissez hole=0)
        color=[1,2,3],  # Ajouter des couleurs distinctes pour chaque zone
        color_discrete_sequence=px.colors.sequential.Cividis
  # Palette de couleurs
        )

   # Afficher le graphique avec Streamlit
        st.plotly_chart(fig)
with col3:
    st.markdown("#### Proba associé par intervalle")


    bins = [0.3,0.4,0.5, 0.6,0.7, 0.8, 0.9,1]  
    labels = [ '0.3-0.4','0.4,0.5','0.5,0.6','0.6-0.7', '0.7-0.8', '0.8-0.9','0.9-1']  # Étiquettes correspondantes

# Ajouter une nouvelle colonne 'intervalle' pour associer chaque probabilité à un intervalle
    df['intervalle'] = pd.cut(df['proba_associée'], bins=bins, labels=labels, right=False)

# Compter le nombre d'occurrences dans chaque intervalle
    counts = df['intervalle'].value_counts().sort_index()
    

# Créer un graphique avec Plotly
    fig = px.bar(
        x=counts.index,
        y=counts.values,
        labels={'x': 'Intervalle de probabilité', 'y': 'Nombre d\'occurrences'},
        #title="Répartition des probabilité par intervalle"
)

# Afficher le graphique
    st.plotly_chart(fig)





