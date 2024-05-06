from .model import Model

import logging
import random
import os
import inspect
import pickle
import numpy as np
from gensim.models import doc2vec

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

base_file_path = inspect.getframeinfo(inspect.currentframe()).filename
base_path = os.path.dirname(os.path.abspath(base_file_path))
project_dir_path = os.path.dirname(os.path.abspath(base_path))
classifiers_path = os.path.join(project_dir_path, 'classifiers')
path='/kaggle/working'

class doc2VecModel(Model):

    def __init__(self):
        super().__init__()

    def initialize_model(self, corpus,config):
        logging.info("Building Doc2Vec vocabulary")
        self.corpus = corpus
        self.config=config
        self.model = doc2vec.Doc2Vec(min_count=5,
                                     # Ignores all words with
                                     # total frequency lower than this
                                     window=5,
                                     # The maximum distance between the current
                                     #  and predicted word within a sentence
                                     vector_size=200,  # Dimensionality of the
                                     #  generated feature vectors
                                     workers=16,  # Number of worker threads to
                                     #  train the model
                                     alpha=0.025,  # The initial learning rate
                                     min_alpha=0.00025,
                                     # Learning rate will linearly drop to
                                     # min_alpha as training progresses
                                     dm=config['dm'],
                                     hs=config['hs'],
                                     negative=config['negative']
                                     )
        # dm defines the training algorithm.
        #  If dm=1 means 'distributed memory' (PV-DM)
        # and dm =0 means 'distributed bag of words' (PV-DBOW)
        self.model.build_vocab(self.corpus)

    def train_model(self):
        logging.info("Training Doc2Vec model")
        # 10 epochs take around 10 minutes on my machine (i7),
        #  if you have more time/computational power make it 20
        for epoch in range(5):
            logging.info('Training iteration #{0}'.format(epoch))
            self.model.train(
                self.corpus, total_examples=self.model.corpus_count,
                epochs=self.model.epochs)
            # shuffle the corpus
            random.shuffle(self.corpus)
            # decrease the learning rate
            self.model.alpha -= 0.0002
            # fix the learning rate, no decay
            self.model.min_alpha = self.model.alpha
        #save
        # string='d2lmodel_'+str(self.config['dm'])+str(self.config['hs'])+str(self.config['negative'])+'.pkl'
        # os.makedirs(os.path.join(path,'model'), exist_ok=True)
        # with open(os.path.join(path,'model',string), 'wb') as file:
        #     pickle.dump(self.model, file)

    def get_vectors(self, corpus_size, vectors_size, vectors_type):
        """
        Get vectors from trained doc2vec model
        :param doc2vec_model: Trained Doc2Vec model
        :param corpus_size: Size of the data
        :param vectors_size: Size of the embedding vectors
        :param vectors_type: Training or Testing vectors
        :return: list of vectors
        """
        vectors = np.zeros((corpus_size, vectors_size))
        for i in range(0, corpus_size):
            prefix = vectors_type + '_' + str(i)
            vectors[i] = self.model.docvecs[prefix]
        return vectors

    def label_sentences(corpus, label_type):
        """
        Gensim's Doc2Vec implementation requires each
         document/paragraph to have a label associated with it.
        We do this by using the LabeledSentence method.
        The format will be "TRAIN_i" or "TEST_i" where "i" is
        a dummy index of the review.
        """
        labeled = []
        for i, v in enumerate(corpus):
            label = label_type + '_' + str(i)
            labeled.append(doc2vec.LabeledSentence(v.split(), [label]))
        return labeled
