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
    def read_termframe_data():
        colls = ["en", "hr", "sl"]
        base_path = "../data/oznaceno/"

        data = {}
        count = {}

        for coll in colls:
            curr_path = base_path + coll
            files = [f for f in listdir(curr_path) if isfile(join(curr_path, f))]
            count[coll] = 0

            for fl in files:
                d = open(f"{curr_path}/{fl}", "r")
                sents = d.read()

        print(count)



