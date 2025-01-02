# -*- coding: utf-8 -*-
"""streamlit.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1U9SdneS60j_sQ_KNVaQFF9aShV9nmCA4
"""



import streamlit as st
import pandas as pd
import altair as alt

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

#zone_filter=st.selectbox("Select a zone",("italie","allemagne","espagne"))
#df = df[df["zone"] == zone_filter]

time_filter=st.selectbox("Choisissez un mois",pd.unique(df["Mois"]))
df=df[df["Mois"]==time_filter]

col = st.columns((1.5, 4.5, 2), gap='medium')

df

fig_col1, fig_col2 = st.columns(2)

with fig_col2:
    st.markdown("### Second Chart")
    fig2 = px.histogram(data_frame=df, x="zone")
    st.write(fig2)

with fig_col2:
    st.markdown("### Second Chart")
    fig2 = px.histogram(data_frame=df, x="zone")
    st.write(fig2)

st.markdown("### Detailed Data View")
st.dataframe(df)





df1













