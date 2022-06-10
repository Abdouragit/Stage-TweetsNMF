import pandas as pd

#Pour la modélisation

#TfidVect permet d'obtenir une matrice de fréquences
from sklearn.feature_extraction.text import TfidfVectorizer

#Nmf decomposition de notre matrice docs X mots 
# en docs X topics et topics X mots
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
#pd.set_option('max_colwidth', 150)


#Chargement du csv
data = pd.read_csv("inaug_speeches.csv", header = 0, encoding = 'latin1', engine = 'python')

#Affichage du tableau
print(data.head())

#On ne garde que les features Name et Speech
data = data.iloc[:, [1, 4] ]

#On ne garde que les lignes correspondant aux premiers mandats (certains presidents n'ont pas fait de 2nd mandat)
data = data.drop_duplicates(subset=['Name'], keep ='first')

#On actualise les index
data.reset_index

#Les index sont mis sur les noms des présidents 
data = data.set_index('Name')

#Fonction de nettoyage

def clean1(text) :
    #met en minuscule, enleve les caractères etranges, enleve les nombres et le texte entre crochets

    #met en minuscule
    text = text.lower()

    #enleve les car chelous
    text = re.sub('ï¿½ï¿½', ' ', text)

    #enleve le texte entre crochet
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ',text)
    
    #enleve les mots qui contiennent des nombres
    text = re.sub('\w*\d\w*', ' ', text)

    return text


round1 = lambda x : clean1(x)

#Nettoyage des speechs
data["text"] = data["text"].apply(round1)
print(data.head())



#Lemmatization et extraction des noms uniquement

def nom(text) :
    #tokenization du texte et extraction des noms

    #Fonction silencieuse de filtrage des noms
    is_noun = lambda pos : pos[:2] == 'NN'

    #tokenisation et stockage dans une listes
    tokenized = word_tokenize(text)

    #stockage d'une fonction
    wordlemmatizer = WordNetLemmatizer()

    #Comprehension de liste pour extraire uniquement les noms
    ToutNoms = [wordlemmatizer.lemmatize(word) for (word, pos) in pos_tag(tokenized) if is_noun(pos)]
    
    #On renvoi un string avec tout les noms
    return ' '.join(ToutNoms)

#Nouveau data frame avec les process

dataNoms = pd.DataFrame(data.text.apply(nom))

print(dataNoms.head())


#Creation d'une matrice Document termes

#On rajoute des stopwords qui sont inutiles
stopnouns = ['america', 'today', 'thing']
allstopwords = text.ENGLISH_STOP_WORDS.union(stopnouns)

#Creation d'une matrice faite uniquement de noms
Mnoms = TfidfVectorizer(stop_words = allstopwords, max_df = .8, min_df = .01)

#On transforme les données pour qu'elles fitent dans la matrice
data_Mnoms = Mnoms.fit_transform(dataNoms.text)

#Creation d'un data frame de la matrice docs X termes

DocTermes = pd.DataFrame(data_Mnoms.toarray(), columns = Mnoms.get_feature_names_out())

DocTermes.index = data.index

print(DocTermes.head())