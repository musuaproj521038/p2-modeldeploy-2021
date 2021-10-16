# Load libraries 
import json
import os 
import spacy
import pickle
import joblib
import subprocess

import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk import pos_tag, pos_tag_sents

from sklearn.preprocessing import MinMaxScaler

import readability

from azureml.contrib.services.aml_request import AMLRequest, rawhttp
from azureml.contrib.services.aml_response import AMLResponse

def readability_measurements(passage: str):
    """
    This function uses the readability library for feature engineering.
    It includes textual statistics, readability scales and metric, and some pos stats
    source: https://www.kaggle.com/ravishah1/readability-feature-engineering-non-nn-baseline/notebook
    """
    results = readability.getmeasures(passage, lang='en')
    
    chars_per_word = results['sentence info']['characters_per_word']
    syll_per_word = results['sentence info']['syll_per_word']
    words_per_sent = results['sentence info']['words_per_sentence']
    
    kincaid = results['readability grades']['Kincaid']
    ari = results['readability grades']['ARI']
    coleman_liau = results['readability grades']['Coleman-Liau']
    flesch = results['readability grades']['FleschReadingEase']
    gunning_fog = results['readability grades']['GunningFogIndex']
    lix = results['readability grades']['LIX']
    smog = results['readability grades']['SMOGIndex']
    rix = results['readability grades']['RIX']
    dale_chall = results['readability grades']['DaleChallIndex']
    
    tobeverb = results['word usage']['tobeverb']
    auxverb = results['word usage']['auxverb']
    conjunction = results['word usage']['conjunction']
    pronoun = results['word usage']['pronoun']
    preposition = results['word usage']['preposition']
    nominalization = results['word usage']['nominalization']
    
    pronoun_b = results['sentence beginnings']['pronoun']
    interrogative = results['sentence beginnings']['interrogative']
    article = results['sentence beginnings']['article']
    subordination = results['sentence beginnings']['subordination']
    conjunction_b = results['sentence beginnings']['conjunction']
    preposition_b = results['sentence beginnings']['preposition']
    
    return [chars_per_word, syll_per_word, words_per_sent,
            kincaid, ari, coleman_liau, flesch, gunning_fog, lix, smog, rix, dale_chall,
            tobeverb, auxverb, conjunction, pronoun, preposition, nominalization,
            pronoun_b, interrogative, article, subordination, conjunction_b, preposition_b]

def pos_tag_features(passage: str):
    """
    This function counts the number of times different parts of speech occur in an excerpt
    """
    pos_tags = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", 
                "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "RB", "RBR", "RBS", "RP", "TO", "UH",
                "VB", "VBD", "VBG", "VBZ", "WDT", "WP", "WRB"]
    
    tags = pos_tag(word_tokenize(passage))
    tag_list= list()
    
    for tag in pos_tags:
        tag_list.append(len([i[0] for i in tags if i[1] == tag]))
    
    return tag_list

def generate_other_features(passage: str):
    # punctuation count
    periods = passage.count(".")
    commas = passage.count(",")
    semis = passage.count(";")
    exclaims = passage.count("!")
    questions = passage.count("?")
    
    # Some other stats
    num_char = len(passage)
    num_words = len(passage.split(" "))
    unique_words = len(set(passage.split(" ") ))
    word_diversity = unique_words/num_words
    
    word_len = [len(w) for w in passage.split(" ")]
    longest_word = np.max(word_len)
    avg_len_word = np.mean(word_len)
    
    return [periods, commas, semis, exclaims, questions,
            num_char, num_words, unique_words, word_diversity,
            longest_word, avg_len_word]

def spacy_features(df: pd.DataFrame):  
    nlp = spacy.load('en_core_web_lg')
    with nlp.disable_pipes():
        vectors = np.array([nlp(text).vector for text in df.excerpt])
        
    return vectors

def get_spacy_col_names():
    names = list()
    for i in range(300):
        names.append(f"spacy_{i}")
        
    return names

features = ["chars_per_word", "syll_per_word", "words_per_sent",
            "kincaid", "ari", "coleman_liau", "flesch", "gunning_fog", "lix", "smog", "rix", "dale_chall",
            "tobeverb", "auxverb", "conjunction", "pronoun", "preposition", "nominalization", 
            "pronoun_b", "interrogative", "article", "subordination", "conjunction_b", "preposition_b"]
features += get_spacy_col_names()
features += ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", 
            "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "RB", "RBR", "RBS", "RP", "TO", "UH",
            "VB", "VBD", "VBG", "VBZ", "WDT", "WP", "WRB"]

class CLRDataset:
    """
    This is my CommonLit Readability Dataset.
    By calling the get_df method on an object of this class,
    you will have a fully feature engineered dataframe
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.excerpts = df["excerpt"]
        
        self._extract_features()
        
        #if train:
            #self.df = create_folds(self.df, n_folds)
        
    def _extract_features(self):
        scores_df = pd.DataFrame(self.df["excerpt"].apply(lambda p : readability_measurements(p)).tolist(), 
                                 columns=["chars_per_word", "syll_per_word", "words_per_sent",
                                          "kincaid", "ari", "coleman_liau", "flesch", "gunning_fog", "lix", "smog", "rix", "dale_chall",
                                          "tobeverb", "auxverb", "conjunction", "pronoun", "preposition", "nominalization",
                                          "pronoun_b", "interrogative", "article", "subordination", "conjunction_b", "preposition_b"])
        self.df = pd.merge(self.df, scores_df, left_index=True, right_index=True)
        
        spacy_df = pd.DataFrame(spacy_features(self.df), columns=get_spacy_col_names())
        self.df = pd.merge(self.df, spacy_df, left_index=True, right_index=True)
        
        pos_df = pd.DataFrame(self.df["excerpt"].apply(lambda p : pos_tag_features(p)).tolist(),
                              columns=["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", 
                                       "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "RB", "RBR", "RBS", "RP", "TO", "UH",
                                       "VB", "VBD", "VBG", "VBZ", "WDT", "WP", "WRB"])
        self.df = pd.merge(self.df, pos_df, left_index=True, right_index=True)
        
        other_df = pd.DataFrame(self.df["excerpt"].apply(lambda p : generate_other_features(p)).tolist(),
                                columns=["periods", "commas", "semis", "exclaims", "questions",
                                         "num_char", "num_words", "unique_words", "word_diversity",
                                         "longest_word", "avg_len_word"])
        self.df = pd.merge(self.df, other_df, left_index=True, right_index=True)
        
    def get_df(self):
        return self.df
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int):
        pass

def set_seed(seed=42):
    """ Sets the Seed """
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


# 1. Requried init function
def init():
    # Create a global variable for loading the model
    global model
    global scaler

    scaler = MinMaxScaler()
    set_seed(42)
    model = []

    # Download needed libraries for spacy features.
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

    # Obtain the needed models.
    import glob
    listModel = glob.glob(os.path.join(os.getenv("AZUREML_MODEL_DIR"), f"*/*/model*.joblib"))

    for i in range(len(listModel)):
        model.append(joblib.load(listModel[i]))

    # Fit the transform.
    train_dataset = CLRDataset(pd.read_csv("source_dir/train.csv")).get_df()
    train_dataset[features] = scaler.fit_transform(train_dataset[features])


    

# 2. Requried run function
@rawhttp
def run(request):
    # Receive the data and run model to get predictions 
    data = json.loads(request.get_data(True))
    try:
        data_formatted = pd.DataFrame([data])
        res = preprocess(data_formatted)
        
        resp = AMLResponse(json.dumps({"id":data["id"], "target": res.tolist()[0]}), 200, json_str=True)
        resp.headers['Access-Control-Allow-Origin'] = "*"
        return resp
    except Error as e:
        return AMLResponse("Bad request", 400)

# Convert the data so the model can process the input as requested.
def preprocess(dataframe: pd.DataFrame):
    data_formatted_dataset = CLRDataset(dataframe).get_df()
    data_formatted_dataset[features] = scaler.transform(data_formatted_dataset[features])

    all_preds = pd.DataFrame()

    for i in range(2):
        all_preds[f"BR_{i}"] = model[i].predict(data_formatted_dataset[features])

    # Average of the two models used and return target result.
    all_preds["BR"] = all_preds.mean(axis=1)
    return all_preds["BR"]
