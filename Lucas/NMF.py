import pandas as pd
from tabulate import tabulate

#Pour la modélisation

#TfidVect permet d'obtenir une matrice de fréquences
from sklearn.feature_extraction.text import TfidfVectorizer

#Nmf decomposition de notre matrice mots X docs 
# en mots X topics et topics X doc
from sklearn.decomposition import NMF as nmf
from sklearn.feature_extraction import text

#pour le prétraitement des documents

from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer

#Pour le nettoyage des données
import re
import string

#Elargissement des tableaux pandas pour la lisibilité
pd.set_option('max_colwidth', 150)


#Chargement du csv
df = pd.read_csv("inaug_speeches.csv",encoding = 'latin1', engine = 'python')

#Affichage du tableau
print(df.head())



#On ne garde que les lignes correspondant aux premiers mandats (certains presidents n'ont pas fait de 2nd mandat)
df = df.drop_duplicates(subset=['Name'], keep ='first')

#On actualise les index
df = df.reset_index

#On ne garde que les features Name et Speech
df = df[['Name', 'text']]

#Les index sont mis sur les noms des présidents 
df = df.set_index('Name')