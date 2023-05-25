import math, joblib, os, warnings, pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from functools import partial
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings("ignore")

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV

import missingno as msno
from sklearn.decomposition import PCA
import streamlit as st

def main():
    # Create a sidebar menu
    st.sidebar.title("Menu")
    menu_options = ["Home", "Data visualization", "Try predictions"]
    
    selected_option = st.sidebar.selectbox("Select Option", menu_options)

    # Display content based on the selected option
    if selected_option == "Home":
        show_home()
    elif selected_option == "Data visualization":
        show_dv()
    elif selected_option == "Try predictions":
        show_tp()
        

def show_home():
    st.title("Home Page")
    st.subheader('This is a webpage to analyse the Online Shoppers Intention Dataset')
    st.write('---')

def show_dv():
    st.markdown("# Data Visualization Page")
    st.markdown("The goal of this page is to have a look at the data, understand it and see what we may do with it")
    
    # Missing Data
    st.markdown("## Missing Data : Is there any ?")
    col1, col2 = st.columns([1,2])
    with col1:
        container = st.container()
        container.markdown("### Count of missing data by columns")
        st.write(data.isnull().sum(axis=0))
    with col2:
        container = st.container()
        container.markdown("### Visualization of the missing data")
        msno.matrix(data)
        container.pyplot(plt.gcf())
    st.markdown("### There is no missing data in this dataset.")
    st.write("##")
    st.write("##")
    st.write("##")
    st.write("##")
    
    # Correlation
    st.markdown("## Corerlation Analysis")
    col1, col2 = st.columns([1.5,1])
    with col1:
        container = st.container()
        plt.subplots(figsize=(15,15))
        matrix = np.tril(data.corr(numeric_only = True))
        sns.heatmap(data.corr(numeric_only = True), annot = True, cmap= 'coolwarm',square=True,mask = matrix)
        container.pyplot(plt.gcf())
    with col2:
        container = st.container()
        container.markdown("There is a high positive correlation between the PageValues and the target (Revenue)")
    

    # PCA 
    st.markdown("## Principal Components Analysis")
    col1, col2 = st.columns([2,1])
    with col1:
        container = st.container()
        with open('artifacts/X_train.pickle', 'rb') as file:
            X_train = pickle.load(file)
        
        pca = PCA(n_components=20)
        X_train_pca = pca.fit_transform(X_train)
        per_var = np.round(pca.explained_variance_ratio_*100,decimals=1)
        labels = [str(x) for x in range(1,len(per_var)+1)]
        plt.rcParams['figure.figsize'] = (20, 10)
        font=20
        fig = px.bar(x = range(1,len(per_var)+1), y = per_var, title = 'Scree Plot', 
                     labels = {'x' : 'Principal Components',
                               'y' : 'Percentage of Explained Variance'})
        fig.update_layout(xaxis=dict(tickmode='linear', dtick=1))
        container.plotly_chart(fig)
    with col2:
        container = st.container()
        container.markdown("We can use principal component analysis to see if we can have a good representation of our dataset in two or three dimensions.")
        container.markdown("Because the percentage of explained variance decay slowly, it is not possible to represent well our dataset in two or three dimensions.")
    
def show_tp():
    st.title("Try Prediction Page")
    # Input values
    col1, col2 = st.columns(2)
    target = 'Revenue'
    for col_name, col_type in data.dtypes.items():
        if col_name != target:
            if col_type == 'object':
                with col1:
                    add_selectbox = st.selectbox(col_name, set(data[col_name].unique().tolist()))
            elif col_type == 'bool':
                with col1:
                    add_selectbox = st.selectbox(col_name, set(data[col_name].unique().tolist()))
            elif len(data[col_name].unique())<=20:
                with col1:
                    add_selectbox = st.selectbox(col_name, set(data[col_name].unique().tolist()))
            else:
                with col2:
                    if data[col_name].max()>1:
                        slider = st.slider(col_name, int(data[col_name].min()), int(data[col_name].max()), 1)
                    else:
                        slider = st.slider(col_name, data[col_name].min(), data[col_name].max(), 0.01)
    
if __name__ == "__main__":
    data = pd.read_csv('online_shoppers_intention.csv')
    st.set_page_config(page_title = "Online Shoppers Intention Data Analysis", page_icon = ":shopping_bags:", layout = 'wide')
    main()
    
    
