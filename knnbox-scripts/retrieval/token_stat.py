from enum import Enum
import math
from typing import List, Literal
from dotenv import load_dotenv
import os

load_dotenv()

def get_data_store():
    return os.getenv('DATASTORE_NAME')
    #return "reduced_ted/01"
    #return "xshort"
def need_update():
    return False

def get_knn_layer():
    return 6

nlp = None

pos_dict = {
    "NOUN": 0,
    "ADJ": 1,
    "VERB": 2,
    "DET": 3,
    "PROPN": 4,
    "ADP": 5,
    "PUNCT": 6,
    "NUM": 7,
    "SCONJ": 8,
    "CCONJ": 9,
    "AUX": 10,
    "CCONJ": 11,
    "X": 12,
}
def pos_to_int(key):
    return pos_dict[key]

class Error(Enum):
    CORRECT = 0
    INCORRECT = 1
    ORDER = 2
    ENDING = 3
    EXTRA = 4
    NAME = 5
    SOURCE = 6
    INCOMPLETE = 7
    def is_correct(self):
        return self == Error.CORRECT
    def int_to_error(value):
        return Error._value2member_map_[value]
    def count():
        return len(Error._value2member_map_)
    def getId(self):
        return self.value

class WmtErrorType(Enum):
    Noerror = 0
    Fluency = 1
    Accuracy = 2
    Style = 3
    Terminology = 4
    Localeconvention = 5
    Nontranslation = 6
    Other = 7
    def is_correct(self):
        return self == WmtErrorType.Noerror
    def int_to_error(value):
        return WmtError._value2member_map_[value]
    def count():
        return len(WmtError._value2member_map_)

class WmtErrorSeverity(Enum):
    noerror = 0
    Neutral = 1
    Minor = 2
    Major = 3
class WmtError:
    def __init__(self, type: str, severity: str) -> None:
        self.type = type
        self.severity = severity
    def is_correct(self):
        return self.type.is_correct()
    def getId(self):
        return self.type.value
class Token:
    def __init__(self):
        self.sum_1 = 0#
        self.sum_5 = 0#
        self.sum_k = 0#
        self.total_correct_predicitions = 0#
        self.at_least_one = 0
        self.precision = []#
        self.recall = []#
        self.avg_recall = 0#
        self.avg_precision = 0#
        self.retrieved_sentences = []#
        self.rec_token_id = []#
        self.cos_sims = []  # Sentence level similarity. Taking the average as final score
        self.avg_cos_sim = 0
        self.max_prob_token = 0#
        self.rec_token_id = 0#
        self.distances = []  # List of knn-distances. Taking the average as final score
        self.annotation = Error.CORRECT
        self.different_count = 0  # Number of different knn-proposals
        self.chosen_token_id = 0  # Model prediction equals retrieved knn-tokens. Check if this one in rec_token_id
        self.chosen_token_prob = 0
        self.pos = 0


class SentenceStat:
    def __init__(self):
        self.sum_1 = 0#
        self.sum_5 = 0#
        self.sum_k = 0#
        self.tgt_count = 0
        self.avg_recall = 0#
        self.avg_precision = 0#
        self.src_str = ""#
        self.tgt_str = "" 
        self.total_avg_cos_sim = 0
        self.avg_most_similar_dist = 0  # Average KNN distance to 1 score
        self.annotation = []
        self.tokens: List[Token] = []#
        self.retriev_eq_most_likely = 0#
        self.annotation_correct = 0#
        self.error_count = []
        self.score = 0
    def sentence_stat_init(self):
        size = len(self.tokens)
        self.sum_1 = sum(map(lambda x: x.sum_1, self.tokens)) / size
        self.sum_5 = sum(map(lambda x: x.sum_5, self.tokens)) / size
        self.sum_k = sum(map(lambda x: x.sum_k, self.tokens)) / size
        avg_recall = math.fsum(map(lambda x: x.sum_k, self.tokens))
        dist_most_sim = list(map(lambda x: x.distances[0], self.tokens))
        self.avg_most_similar_dist = math.fsum(dist_most_sim) / len(dist_most_sim)
        self.retriev_eq_most_likely = sum(map(lambda x: x.max_prob_token == x.rec_token_id[0] , self.tokens)).item() / size
        self.annotation_correct = sum(map(lambda x: x.annotation.is_correct(), self.tokens)) / size
        self.total_avg_cos_sim = math.fsum(map(lambda x: x.avg_cos_sim, self.tokens)) / size
        self.error_count = [0] * Error.count()
        i = 0
        for x in self.tokens:
            self.error_count[x.annotation.getId()] += 1
            i += 1
        for i in range(0, len(self.error_count)):
            self.error_count[i] /= len(self.tokens)
        self.knn_store_layer = get_knn_layer()
    def add_pos(self, dict):
        global nlp
        if nlp == None:
            import spacy
            import de_core_news_sm
            nlp = spacy.load("de_core_news_sm")
            nlp = de_core_news_sm.load()

        doc = nlp(self.tgt_str)
        tags = [(w.text, w.pos_) for w in doc]
        def skip_foward(current, i):
            while not current.isalpha():
                #print("Skipped", current, "in words")
                if i < len(tags) - 1:
                    i += 1
                    current = tags[i][0]
                else:
                    break
            return current, i
        i = 0
        current = tags[i][0]
        l = len(current)
        for token in self.tokens:
            t = token.chosen_token_id
            t_str = dict[t].replace("▁", "")
            #print(t_str)
            if not t_str.isalpha():
                #print("Skipped", t_str, "in token")
                current, i = skip_foward(current, i)
                l = len(current)
                token.pos = "PUNC"
                continue
            l -= len(t_str)
            #print("Len:", l)
            #print(t_str, ", ", tags[i][1])
            token.pos = tags[i][1]
            if l == 0 and i < len(tags)-1:
                i += 1
                current = tags[i][0]
                l = len(current)
        self.tokens[-1].pos = "X"
    def add_annotation(self, annotation):
        for i, t in enumerate(self.tokens):
            t.annotation = annotation[i]

    def sort_by_error(self):
        tag_dict = {}
        for token in self.tokens:
            if not token.annotation in tag_dict:
                tag_dict[token.annotation] = SentenceStat()
            tag_dict[token.annotation].tokens.append(token)
        for sentence in tag_dict.values():
            sentence.sentence_stat_init()
        return tag_dict
    def sort_by_pos(self):
        tag_dict = {}
        for token in self.tokens:
            if not token.pos in tag_dict:
                tag_dict[token.pos] = SentenceStat()
            tag_dict[token.pos].tokens.append(token)
        for sentence in tag_dict.values():
            sentence.sentence_stat_init()
        return tag_dict
