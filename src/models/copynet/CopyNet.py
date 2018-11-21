import tensorflow as tf
import numpy as np

from hbconfig import Config
from models.copynet.DataLoader import DataLoader
from models.copynet.Generator import Generator
from models.attention.Attention import Attention
from utils.Seq2SeqHelper import *

class CopyNet(Attention):
    def __init__(self):
        super(CopyNet,self).__init__() 
        Config('configs/en_word.yml')
        self._dl = DataLoader()
        self._g = Generator(self._dl)