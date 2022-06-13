# Bienvenue dans cette tentative de faire une NMF et + si affinité

# ---- Importation des modules 

import pandas as pd #on en aura besoin pour manipuler des dataframes mais aussi pour importer/exporter des données

from sklearn.feature_extraction import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.feature_extraction import text
# sklearn est surement une bibliothèque remplie de modules et chaque module est remplie de classes qu'on importe
# sklearn sert à faire de la modélisation

from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
""" Alors, je ne sais pas exactement à quoi sert chaque élément
ci-dessus. Cependant on a besoin des toutes ces classes et fonctions
afin de faire du "text processing". Cet-à-dire que l'on va faire
des modifications textuelles avec.
"""

import re
import string
# for text cleaning

# ---- Importation des données

pd.set_option('max_colwidth',150)

df = pd.read_csv('inaug_speeches.csv',engine = 'python')

df.head()




