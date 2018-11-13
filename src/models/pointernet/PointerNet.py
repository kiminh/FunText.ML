import tensorflow as tf
import numpy as np

from hbconfig import Config
from models.pointernet.PointerNetDataLoader import DataLoader
from models.pointernet.PointerNetGenerator import Generator
from models.seq2seq.Seq2SeqHelper import *

class PointerNet():
    def __init__(self):
        Config('configs/letter.yml')

    def train(self):
        dl = DataLoader()
        g = Generator(dl)

        train_input_fn = lambda: dl.make_batch(dl.X_train, dl.y_train)
        test_input_fn = lambda: dl.make_batch(dl.X_test, dl.y_test)
        
        estimator = tf.estimator.Estimator(g.model_fn, model_dir=Config.data.out_dir+"checkpoint")
        
        for _ in range(Config.train.num_epochs):
            estimator.train(train_input_fn)
            estimator.evaluate(test_input_fn)
            
        with open(Config.data.predict_prefix+Config.data.src_suffix, 'r', encoding='utf-8') as f:
            text = f.read()            
        s = g.predict_fn(text, estimator)
        with open(Config.data.predict_prefix+Config.data.tgt_suffix, "w", encoding="utf-8") as f:
            f.write(s)

    def predict(self):
        dl = DataLoader()
        g = Generator(dl)
        
        estimator = tf.estimator.Estimator(g.model_fn, model_dir=Config.data.out_dir+"checkpoint")

        with open(Config.data.predict_prefix+Config.data.src_suffix, 'r', encoding='utf-8') as f:
            text = f.read()            
        s = g.predict_fn(text, estimator)
        with open(Config.data.predict_prefix+Config.data.tgt_suffix, "w", encoding="utf-8") as f:
            f.write(s)