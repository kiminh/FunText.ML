import tensorflow as tf
import numpy as np

from hbconfig import Config
    
PAD,UNK,SOS,EOS = '<PAD>','<UNK>','<SOS>','<EOS>'
PAD_ID,UNK_ID,SOS_ID,EOS_ID = 0,1,2,3

def read_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()
        
def build_map(words):
    specials = [PAD,UNK,SOS,EOS]
    idx2word = {idx: word for idx, word in enumerate(specials + words)}
    word2idx = {word: idx for idx, word in idx2word.items()}
    return idx2word, word2idx
            
def load_embed(embed_file):    
    vecs = []
    words = []
    is_first_line = True
    with open(embed_file, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.rstrip().split(" ")
            if is_first_line:
                is_first_line = False
                continue
            word = tokens[0]
            vec = list(map(float, tokens[1:]))
            vecs.append(vec)
            words.append(word)
    word_embs = np.array(vecs, dtype=np.float32)
    return words,word_embs

def select_by_score(predictions):
    p_list = list(predictions)

    scores = []
    for p in p_list:
        score = 0
        unknown_count = len(list(filter(lambda x: x == -1, p)))
        score -= 2 * unknown_count
        eos_except_last_count = len(list(filter(lambda x: x == EOS_ID, p[:-1])))
        score -= 2 * eos_except_last_count
        distinct_id_count = len(list(set(p)))
        score += 1 * distinct_id_count
        if eos_except_last_count == 0 and p[-1] == EOS_ID:
            score += 5
        scores.append(score)

    max_score_index = scores.index(max(scores))
    return predictions[max_score_index]

def clip_grads(loss):
    variables = tf.trainable_variables()
    grads = tf.gradients(loss, variables)
    clipped_grads, _ = tf.clip_by_global_norm(grads, Config.model.max_gradient_norm)
    return zip(clipped_grads, variables)

def create_rnn_cell(num_layers, mode):
    cell_list = _cell_list(num_layers, mode)
    if len(cell_list) == 1:
        return cells[0]
    else:
        return tf.contrib.rnn.MultiRNNCell(cell_list)

def _cell_list(num_layers, mode):
    single_cell_fn = _single_cell
    cell_list = []
    for i in range(num_layers):
        single_cell = single_cell_fn(mode)
        cell_list.append(single_cell)

    return cell_list

def _single_cell(mode):
    single_cell = tf.nn.rnn_cell.GRUCell(Config.model.num_units)
    
    dropout = Config.model.dropout if mode == tf.estimator.ModeKeys.TRAIN else 0.0
    if dropout > 0.0:
        single_cell = tf.contrib.rnn.DropoutWrapper(
            cell=single_cell, input_keep_prob=(1.0 - dropout))

    return single_cell
        