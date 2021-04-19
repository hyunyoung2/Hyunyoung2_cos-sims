#-*- coding: utf-8 -*-                                                                                                                                                               
"""
This code is to caculate similarity between two documents


First of all, this code extracts noun sequence from a line 


Second, this code calculate TFIDFVectors

""" 

from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 


DEBUG_LEVEL0 = "debug_level_0" ## No Debug

DEBUG_LEVEL1 = "debug_level_1" ## easy
DEBUG_LEVEL2 = "debug_level_2" ## middle
DEBUG_LEVEL3 = "debug_level_3" ## hard

DEBUG_OPTION = DEBUG_LEVEL1


RAW_CORPUS = "data/news1000-utf8.txt" 

def read_raw_corpus(path):
    """This function reads raw corpus 

    we deal with the white-space token as a particular token

    Args: 
        path(string): file location

    Return:
        data(list): data lines 
    """

    with open(path, "r") as rf:
        data = [val.strip() for val in rf.readlines() if val != "\n"]
        
    if DEBUG_OPTION in [DEBUG_LEVEL1, DEBUG_LEVEL2, DEBUG_LEVEL3]:
        print("\n\n===== Reading {} =====".format(path))
        print("\nThe number of lines: {}".format(len(data)))
        print("\nThe top 3 lines:\n{}".format(data[0:3]))

    return data



def make_noun_sequence(txt):
    """This fucntion extracts noun sequences 


    To get noun from each line, we use Okt mopheme analyzer

    for Okt analyzer, refer to https://konlpy.org/en/latest/api/konlpy.tag/

    Args(list): tesx line by line
   
    Returns(list): nouns sequence for each line
    """ 

    okt = Okt()


    noun_data = []
    for idx, val in enumerate(txt):

        # To normalize text
        temp = okt.normalize(val)
        if idx == 0 and DEBUG_OPTION in [DEBUG_LEVEL1, DEBUG_LEVEL2, DEBUG_LEVEL3]:
            print("\n\n===== The normalized text =====")
            print("The previous: {}".format(val)) 
            print("The result: {}".format(temp))

        # To extract nouns
        noun_data.append(" ".join(okt.nouns(temp)))
        
    if DEBUG_OPTION in [DEBUG_LEVEL1, DEBUG_LEVEL2, DEBUG_LEVEL3]:
        print("\n===== The result of extracting nouns =====")
        print("\nThe number of lines: {}".format(len(noun_data)))
        print("\nThe top 3 lines:\n{}".format(noun_data[0:3]))

    return noun_data

def make_tfidfvector(noun_seq):
    """This fucntion calculates tf-idf vector 

    for how to make tf-idf vectore, 
    
    refer to https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

    Args(list): noun sequence for each line 
   
    Returns(list): tf-idf vector for each line
    """ 

    vectorizer = TfidfVectorizer()

    _x = vectorizer.fit_transform(noun_seq)

    x = _x.toarray()  

    if DEBUG_OPTION in [DEBUG_LEVEL1, DEBUG_LEVEL2, DEBUG_LEVEL3]:
        print("\n\n===== The result of tf-idf vectors =====")
        features = vectorizer.get_feature_names()
        print("\nThe number of features: {}".format(len(features)))
        print("\nThe top 10 features: {}".format(features[0:10]))
        
        print("\nThe shape of tf-idf(_x) vector: {}".format(_x.shape))
        print("\nThe type of tf-idf(_x) vector: {}".format(type(_x)))
        print("\nThe top 2 feature(_x) vectors:\n{}".format(_x[0:2]))

        print("\nThe shape of tf-idf(x) vector: {}".format(x.shape))
        print("\nThe length of tf-idf(x) vector: {}".format(len(x)))
        print("\nThe type of tf-idf(x) vector: {}".format(type(x)))
        print("\nThe top 2 of feature(x) vectors:\n{}".format(x[0:2]))

    return _x



def calculate_sims(vecs):
    """This fucntion calculates similarity between two documents


    for cosine similarity of sklearn, 
       refer to https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
   
    Args(ndarray): tf-idf vectors 
    """ 

    sim_matrix = cosine_similarity(vecs)  

    i_matrix = np.identity(vecs.shape[0])*2

    _sims = sim_matrix - i_matrix

    for i in range(0, len(sim_matrix)):
        print("\nMax. sim of doc", i , "is doc", _sims[i].argmax(), "-- cos-sim is", _sims[i].max())
        
        
    if DEBUG_OPTION in [DEBUG_LEVEL1, DEBUG_LEVEL2, DEBUG_LEVEL3]:
        print("\n\n===== The result of similarity between two document =====")
        print("\nThe type of sim_matrix: {}".format(type(sim_matrix)))
        print("\nThe length of sim_matrix: {}".format(len(sim_matrix)))
        print("\nThe shape of sim_matrix: {}".format(sim_matrix.shape))
        print("\nThe top 2 of sim_matrix vectors:\n{}".format(sim_matrix[0:2]))

        print("\nThe type of i_matrix: {}".format(type(i_matrix)))
        print("\nThe length of i_matrix: {}".format(len(i_matrix)))
        print("\nThe shape of i_matrix: {}".format(i_matrix.shape))
        print("\nThe top 2 of i_matrix vectors:\n{}".format(i_matrix[0:2]))

        print("\nThe type of _sims: {}".format(type(_sims)))
        print("\nThe length of _sims: {}".format(len(_sims)))
        print("\nThe shape of _sims: {}".format(_sims.shape))
        print("\nThe top 2 of _sims vectors:\n{}".format(_sims[0:2]))



if __name__ == "__main__":



   new_lines = read_raw_corpus(RAW_CORPUS)

   noun_seqs = make_noun_sequence(new_lines)

   input_x = make_tfidfvector(noun_seqs)

   calculate_sims(input_x)
