import getopt
import sys
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

from colorama import Fore
from models.attention.Attention import Attention
from models.copynet.CopyNet import CopyNet

def set_model(model_name):
    models = dict()
    models['attention'] = Attention
    models['copynet'] = CopyNet
    try:
        Model = models[model_name.lower()]
        model = Model()
        return model
    except KeyError:
        print('Unsupported Model type: ' + model_name)
        sys.exit(-2)

def parse_cmd(argv):
    try:
        opts, args = getopt.getopt(argv, "hg:pt")
        opt_arg = dict(opts)
        if '-h' in opt_arg.keys():
            print('usage: python main.py -g <model_type>')
            print('       python main.py -g <model_type> -p')
            sys.exit(0)
        if not '-g' in opt_arg.keys():
            model = set_model('attention')
        else:
            model = set_model(opt_arg['-g'])
        if '-p' in opt_arg.keys():
            model.predict()
        elif '-t' in opt_arg.keys():
            model.test()
        else:
            model.train()
    except getopt.GetoptError:
        print('invalid arguments!')
        print('`python main.py -h`  for help')
        sys.exit(-1)
    pass

if __name__ == '__main__':
    model = None
    parse_cmd(sys.argv[1:])
