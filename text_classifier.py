from models.doc2vec_model import doc2VecModel
from models.classifier_model import classifierModel

import os
import logging
import inspect

import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

base_file_path = inspect.getframeinfo(inspect.currentframe()).filename
project_dir_path = os.path.dirname(os.path.abspath(base_file_path))
data_path = os.path.join(project_dir_path, 'data')
default_classifier = os.path.join(
    project_dir_path, 'classifiers', 'logreg_model.pkl')
default_doc2vec = os.path.join(project_dir_path, 'classifiers', 'd2v.model')
default_dataset = os.path.join(data_path, 'dataset.csv')
train_dataset= os.path.join(data_path, 'train.parquet')
test_dataset= os.path.join(data_path, 'test.parquet')

class TextClassifier():

    def __init__(self):
        super().__init__()
        self.d2v = doc2VecModel()
        self.classifier = classifierModel()
        self.dataset = None

    def read_data(self, filename):
        filename = os.path.join(data_path, filename)
        self.dataset = pd.read_csv(filename, header=0, delimiter="\t")

    def prepare_all_data(self):
        # x_train, x_test, y_train, y_test = train_test_split(
            # self.dataset.review, self.dataset.sentiment, random_state=0,
            # test_size=0.1)
        x_train = pd.read_parquet(train_dataset)
        y_train= x_train.label
        x_train = x_train.text
        x_test = pd.read_parquet(test_dataset)
        y_test = x_test.label
        x_test = x_test.text
        x_train = doc2VecModel.label_sentences(x_train, 'Train')
        x_test = doc2VecModel.label_sentences(x_test, 'Test')
        all_data = x_train + x_test
        return x_train, x_test, y_train, y_test, all_data

    def prepare_test_data(self, sentence):
        x_test = doc2VecModel.label_sentences(sentence, 'Test')
        return x_test

    def train_classifier(self,config):
        x_train, x_test, y_train, y_test, all_data = self.prepare_all_data()
        self.d2v.initialize_model(all_data,config)
        self.d2v.train_model()
        self.classifier.initialize_model()
        self.classifier.train_model(self.d2v, x_train, y_train,config)
        self.classifier.test_model(self.d2v, x_test, y_test,config)
        return self.d2v, self.classifier

    def test_classifier(self,config):
        _, x_test, _, y_test, _ = self.prepare_all_data()
        if (self.d2v.model is None or self.classifier.model is None):
            logging.info(
                "Models Not Found, Train First or Use Correct Model Names")
        else:
            self.classifier.test_model(self.d2v, x_test, y_test,config)


def run():
    tc = TextClassifier()
    # (1) HS + PV-DM:(2) HS + PV-DBOW:(3) NS + PV-DM:(4) NS + PV-DBOW
    print("HS+PV-DM")
    config={'dm':1,'hs':1,"negative":0}
    tc.train_classifier(config)
    # tc.test_classifier(config)
    print("HS+PV-DBOW")
    config={'dm':0,'hs':1,'negative':0}
    tc.train_classifier(config)
    # tc.test_classifier(config)
    print("NS+PV-DM")
    config={'dm':1,'hs':0,'negative':5}
    tc.train_classifier(config)
    # tc.test_classifier(config)
    print("NS+PV-DBOW")
    config={'dm':0,'hs':0,'negative':5}
    tc.train_classifier(config)
    # tc.test_classifier(config)


    


if __name__ == "__main__":
    run()
