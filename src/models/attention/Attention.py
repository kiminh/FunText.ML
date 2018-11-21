import tensorflow as tf
import numpy as np

from hbconfig import Config
from models.attention.DataLoader import DataLoader
from models.attention.Generator import Generator
from utils.Seq2SeqHelper import *

class Attention():
    def __init__(self):
        Config('configs/en_word.yml')
        self._dl = DataLoader()
        self._g = Generator(self._dl)

    def train(self):
        train_input_fn = lambda: self._dl.get_batch(self._dl.X_train, self._dl.y_train)
        test_input_fn = lambda: self._dl.get_batch(self._dl.X_test, self._dl.y_test,scope='test')
        
        estimator = tf.estimator.Estimator(self._g.model_fn, model_dir=Config.data.out_dir+"checkpoint")
        
        for _ in range(Config.train.num_epochs):
            estimator.train(train_input_fn)
            estimator.evaluate(test_input_fn)
            
        with open(Config.data.predict_prefix+Config.data.src_suffix, 'r', encoding='utf-8') as f:
            text = f.read()            
        s = self._g.predict_fn(text, estimator)
        with open(Config.data.predict_prefix+Config.data.tgt_suffix, "w", encoding="utf-8") as f:
            f.write(s)

    def predict(self):
        estimator = tf.estimator.Estimator(self._g.model_fn, model_dir=Config.data.out_dir+"checkpoint")

        with open(Config.data.predict_prefix+Config.data.src_suffix, 'r', encoding='utf-8') as f:
            text = f.read()            
        s = self._g.predict_fn(text, estimator)
        with open(Config.data.predict_prefix+Config.data.tgt_suffix, "w", encoding="utf-8") as f:
            f.write(s)
