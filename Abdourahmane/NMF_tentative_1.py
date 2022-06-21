# Bienvenue dans cette tentative de faire une NMF et + si affinité

# ---- Importation des modules 

from numpy import argsort
import pandas as pd #on en aura besoin pour manipuler des dataframes mais aussi pour importer/exporter des données

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.feature_extraction import text
# sklearn est surement une bibliothèque remplie de modules et chaque module est remplie de classes qu'on importe
# sklearn sert à faire de la modélisation

from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import nltk
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

    # store dunction to lemmatize each word
    wordnet_lemmatizer = WordNetLemmatizer()

    # use liste comprehension to lemmatize all words
    # and create a list of all nouns
    all_nouns = [wordnet_lemmatizer.lemmatize(word) for (word, pos) in pos_tag(tokenized) if is_noun(pos)]
    
    # return string of joinded list of nouns
    return ''.join(all_nouns)


# creer un dataframe de noms venant des discours uniquement
data_nouns = pd.DataFrame(df.text.apply(nouns))

print()
print("--c'est quoi nouns--")
print(type(nouns))

#Visually inspect
print()
print(data_nouns.head())

    
"""
On dirait qu'on a enlevé toute la ponctuation, les majuscules ainsi que
 les espaces. Enelver les espaces m'a l'air problématique tout de même,
 comment distinguer les mots désormais ?
"""



# ---- Creer la matrice document-mots (mots par document)

"""Here I added some stopwords to the stopword list so we don’t
 get words like ‘America’ in the topics as that is not super
  meaningful in this context. I also use TF-IDF Vectorizer 
  rather than a simple Count Vectorizer in order to give 
  greater value to more unique terms. """

# add additional stop words since we are recreating the document-term matrix
stop_noun = ["america", 'today', 'thing']
stop_words_noun_agg = text.ENGLISH_STOP_WORDS.union(stop_noun)

# create a document-term matrix with only nouns

# store TF-IDF Vectorizer
tv_noun = TfidfVectorizer(stop_words = stop_words_noun_agg,ngram_range= (1,1),max_df=.8, min_df= .01)

# fit and transform speech noun text to a TF-IDF doc-term Matrix
data_tv_noun = tv_noun.fit_transform(data_nouns.text)

# Create data-drame of Doc-term matrix with nouns as column names
data_dtm_noun = pd.DataFrame(data_tv_noun.toarray(), columns = tv_noun.get_feature_names())

# set president's Names as Index
data_dtm_noun.index = df.index

# visually inspect document-term Matrix
print()
print(data_dtm_noun.head())

"""On dirait qu'on a isolé les noms des présidents"""


# ---- Create function to display Topics

"""To evaluate how useful the topics created by NMF are, 
we need to know what they are. Here I create a function to display 
the top words activated for each topic."""

"""
Je crois qu'on essaie là de savoir quels sont les mots les plus 
présents dans les topics qu'on a créé.
"""

def display_topics(model, feature_names, num_top_words, topic_names = None) :
    ''' Given an NMF model, feature_names, and number of top words,
     print topic number and its top feature names, up to specified 
     number of top words.'''

    # iterate through topics in topic-term matrix, 'H' aka
    # model.component_
    for ix, topics in enumerate(model.components_):
        #print topic, topic number, and top words
        if not topic_names or not topic_names[ix]:
            print ("\nTopic ", ix)
        else:
            print("\nTopic: '", topic_names[ix],"'")
        
        print (", ".join([feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]))

