import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(layout='wide')

@st.cache_data
def load_data():
    data = pd.read_csv('BeijingPM.csv', index_col='No')
    data['season'] = data['season'].replace({1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'})
    year_range = data['year'].unique()
    month_range = data['month'].unique()
    season_range = data['season'].unique()

    return data, year_range, month_range, season_range

def groupby_data(data:pd.DataFrame, name:str or list, columns:list) -> pd.DataFrame:
    data = data.groupby(name)[columns].mean().reset_index()
    return data

df, year_range, month_range, season_range = load_data()

st.title('Beijing PM2.5')

year = st.selectbox('Select Year', year_range)

max_PM = df[df.year == year].groupby('month')[['PM_Dongsi', 'PM_Dongsihuan', 'PM_Nongzhanguan', 'PM_US Post']].mean().agg('max')

st.write(f'''
### {year} Beijing PM2.5
''')

m1, m2, m3, m4 = st.columns([1,1,1,1])
m1.metric('The max PM2.5 of ' + max_PM.index[0] + 'is', int(max_PM.iloc[0]))
m2.metric('The max PM2.5 of ' + max_PM.index[1] + 'is', int(max_PM.iloc[1]))
m3.metric('The max PM2.5 of ' + max_PM.index[2] + 'is', int(max_PM.iloc[2]))
m4.metric('The max PM2.5 of ' + max_PM.index[3] + 'is', int(max_PM.iloc[3]))

col1, col2 = st.columns((1, 1))

month_group = groupby_data(df[df.year == year], 'month', ['PM_Dongsi', 'PM_Dongsihuan', 'PM_Nongzhanguan', 'PM_US Post'])
fig = px.line(month_group, x='month', y=['PM_Dongsi', 'PM_Dongsihuan', 'PM_Nongzhanguan', 'PM_US Post'],
             template='seaborn')
fig.update_layout(width=600, height=400, legend=dict(orientation='h', yanchor='bottom',
                                                     y=1.02, xanchor='right', x=1), xaxis_title = 'PM2.5',)
col1.plotly_chart(fig, use_container_width=True)

Iws_data =groupby_data(df[df.year == year], 'month',['Iws'])
fig1 = px.line(Iws_data, x='month', y='Iws')
fig1.update_layout(width=600, height=400, xaxis_title = 'Wind speed (m/s)', yaxis_title = 'value')
col2.plotly_chart(fig1, use_container_width=True)

temp_data = groupby_data(df[df.year == year], ['month', 'day'], ['TEMP'])
temp_data = temp_data.pivot(index='month', columns='day', values='TEMP')
fig3 = px.imshow(temp_data, color_continuous_scale='Hot')
fig3.update_layout(title_text='Temperatures in ' + str(year), title_x=0.5)
st.plotly_chart(fig3, use_container_width=True)