import pandas as pd
import json 
from os import listdir
from os.path import isfile, join
import classla


class Data:
    def __init__(self, path = ""):
        self.dir = path
    
    def read_csv(self, batch_num):
        """ Reads CSV data

        Args:
            batch_num (integer): Batch size of data to consider
        """
        self.data = pd.read_csv(self.dir, limit = batch_num)

    def process(self, pipeline):
        """ Chains operations on the data collection e.g [stopword removal, stemmer, lemmatizer]

        Args:
            pipeline (list): List of transformer functions
        """
        data = self.data

        for transform in pipeline:
            data = transform(data)
        
    def train_test_split():
        """Split data into training and testing set accordingly.
        """

    @staticmethod
    def _map_relation_to_pair(relation, e1, e2):
        pair =  None

    @staticmethod
    def _process_wcl_sentence(sentence):
        
        sent = sentence.split("<RGET>")[1]
        return sent

    def read_wcl_data(self):
        lines = []
        data_dir = "../data/wcl_datasets_v1.2/wikipedia/wiki_good.txt"

        with open(data_dir, 'r') as f:
            lines = f.read().split("# ")

        data = []
        
        for line in lines[1:]:
            data.append(self._process_wcl_sentence(line))

        return data


