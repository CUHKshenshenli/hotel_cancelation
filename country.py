from models import lasso, neural_network
from load_data import load, load_dataset
from country_information import GDP, continent, full_name

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA

import openai
from openai import OpenAI

'''We have many methods to deal with this feature.
   Method 1: Since only specific countries have a significant impact on "is_cancel," 
    we are currently not focusing on the specific countries. 
    Instead, we are only concerned with whether it originates from these specific countries.
   Method 2: We believe that the geographical location of the country and its GDP are the primary factors contributing to tourist cancellations. 
    Therefore, we remove the "country" feature and replace it with the continent they belong to and the GDP. 
    In fact, adding the number of flights would be more reasonable, but the work load becomes too large.
   Method 3: Word Embedding + Clustering'''

#Method 1:
#Here we have already gotten the most important 4 countries by decision tree.
#They are PRT, ESP, DEU, FRA.
def selection_specific_countries(filename: str, new_filename: str, number: int) -> None:
    name = ['PRT', 'ESP', 'DEU', 'FRA']
    data = pd.read_csv(filename)
    for i in range(number):
        Country = name[i]
        data[Country] = data['country'] == Country
    data = data.drop('country', axis=1)
    data.to_csv(new_filename, index = False) 

#Method 2
#We use GDP and continent to replace country
#The information of GDP and continent are stored in file 'country_information.py'
#Mode == 0 -> only keep continent
#Mode == 1 -> only keep GDP
#Mode == 2 -> keep both
def selection_information(filename: str, new_filename: str, mode: int) -> None:
    data = pd.read_csv(filename)
    data = data[data['country'] != 'ATA']
    data = data.reset_index(drop=True)
    country_continents = continent()
    gdp_2015 = GDP()

    if mode == 0:
        for i in range(data.shape[0]):
            data.loc[i, 'continent'] = country_continents[data.loc[i, 'country']]
    elif mode == 1:
        for i in range(data.shape[0]):
            data.loc[i, 'gdp'] = gdp_2015[data.loc[i, 'country']]
    elif mode == 2:
        for i in range(data.shape[0]):
            data.loc[i, 'continent'] = country_continents[data.loc[i, 'country']]
            data.loc[i, 'gdp'] = gdp_2015[data.loc[i, 'country']]
    data = data.drop('country', axis=1)
    data.to_csv(new_filename, index = False) 

#Method 3
#As GloVe does not have some country names, we use GPT
openai.api_key = "Use your own OpenAI API KEY"
client = OpenAI(api_key=openai.api_key)

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def selection_embedding() -> np.array:
    country_full_names = full_name()
    country_emb = {}
    for coun in country_full_names.keys():
        country_emb[coun] = get_embedding(country_full_names[coun], model="text-embedding-ada-002")
    X = np.array(list(country_emb.values()))
    return X

#We use KernelPCA to do dimension reduction
def elbow_plot(kernel: str) -> None:
    kernel_pca = []
    color = ['blue','red','green']
    X = selection_embedding()
    for n in range(1, 4):
        pca = KernelPCA(n_components=n, kernel=kernel)
        X_pca = pca.fit_transform(X)
        wcss = []
        for k in range(1,11):
            kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
            kmeans.fit(X_pca)
            wcss.append(kmeans.inertia_)
    kernel_pca.append(wcss)
    plt.figure(figsize=(10,6))
    plt.grid()
    for i, wcss in enumerate(kernel_pca):
        plt.plot(range(1,11),wcss, linewidth=2, color = color[i], marker ="8")
    plt.xlabel("Values of K")
    plt.xticks(np.arange(1,11,1))
    plt.ylabel("WCSS")
    plt.legend()
    plt.show()

'''
We can choose the best dimension and number of clusters by the following:
elbow_plot('rbf')
And we find that when dimension == 2, number of clusters == 5
'''
def country_clustering(filename: str, new_filename: str, n: int, k: int, kernel: str) -> None:
    data = pd.read_csv(filename)
    X = selection_embedding()
    country_full_names = full_name()
    pca = KernelPCA(n_components=n, kernel=kernel)
    X_pca = pca.fit_transform(X)
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_pca)
    y_kmeans = kmeans.fit_predict(X_pca)
    country_label = {}
    for i, coun in enumerate(country_full_names.keys()):
        country_label[coun] = y_kmeans[i]
    for i in range(data.shape[0]):
        data.loc[i,'country_label'] = country_label[data.loc[i,'country']]
    data = data.drop('country', axis=1)
    data.to_csv(new_filename, index = False) 



