# Bienvenue dans cette tentative de faire une NMF et + si affinité

# ---- Importation des modules 

import pandas as pd #on en aura besoin pour manipuler des dataframes mais aussi pour importer/exporter des données

from sklearn.feature_extraction.text import TfidfVectorizer
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

df = pd.read_csv('Abdourahmane\inaug_speeches.csv',engine = 'python')

print()
print(df.head()) # afin de voir les première lignes de donnée

# ---- Isoler les données

"""I’m going to make a dataframe of the President’s names and 
speeches, and isolate to the first term inauguration speech, 
because some President’s did not have a second term. This makes 
for a corpus of documents with similar lengths."""

df = df.drop_duplicates(subset=['Name'],keep='first')

df = df.reset_index() #je ne sais pas ce qu'est l'index

df = df[['Name','text']]

df = df.set_index('Name')

print()
print (df.head())


# ---- Nettoyage des données

"""I want to make all of the text in the speeches as comparable 
as possible so I will create a cleaning function that removes 
punctuation, capitalization, numbers, and strange characters. 

I use regular expressions for this, which offers a lot of ways 
to ‘substitute’ text. 

There are tons of regular expression cheat sheets, like this one. 
I ‘apply’ this function to my speech column"""

def clean_text_round1(text): # je ne sais pas vraiment ce qu'on fait dans cette fonction
    '''Make text loweracse, remove text in square brackets, 
    remove punctuation, remove read errors, and remove words
    containing numbers.'''

    text = text.lower()
    text = re.sub('\[.*?\]',' ',text)
    text = re.sub('[%s]' % re.escape(string.punctuation),' ',text)
    text = re.sub('\w*\d\w*',' ',text)
    text = re.sub('�',' ', text)

    return text

round1 = lambda x: clean_text_round1(x)

# Clean Speech Text
df["text"] = df["text"].apply(round1)

print()
print(df.head())


# ---- Creer une fonction qui va "pre-process" les données
# ou un truc du genre

#Je crois qu'on va faire des transformations dans le texte et remplacer des mots par d'autres

def nouns (text):
    '''Given a string of text, tokenize the text and pull out 
    only nouns'''

    #create mask to isolate words that are nouns
    is_noun = lambda pos: pos[:2] == 'NN'

    #store function to split string of words
    #into a list of words (tokens)
    tokenized = word_tokenize(text)

    



