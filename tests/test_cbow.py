from nose.tools import with_setup, ok_, eq_, assert_almost_equal, nottest, assert_not_equal
from oswegonlp.constants import * 
from oswegonlp import cbow, most_common, preprocessing
import torch

def setup():
    global word_to_ix, tag_to_ix, X_tr, Y_tr, model, embedding_dim
    
    vocab, word_to_ix = most_common.get_word_to_ix(TRAIN_FILE, max_size=6500)
                
    X_tr = []
    Y_tr = []
    for i,(words,tags) in enumerate(preprocessing.conll_seq_generator(TRAIN_FILE)):
        X_tr.append(words)
        Y_tr.append(tags)
    
    torch.manual_seed(765);
    
    embedding_dim=30
    hidden_dim=30
    model = cbow.CBOW(len(vocab), embedding_dim)

#6.1
def test_build_context():
    cv = cbow.build_context([X_tr[5]], context_size=2)
    
    eq_(len(cv), 9)
    eq_(cv[0][0], ['The', 'third', 'being', 'run'])
    eq_(cv[0][1], 'was')
    eq_(cv[8][0], ['of', 'an', 'firm', '.'])
    eq_(cv[8][1], 'investment')

#6.2
def test_cbow_init():
    global model, embedding_dim, word_to_ix

    eq_(model.embeddings.embedding_dim, embedding_dim)
    eq_(model.embeddings.num_embeddings, len(word_to_ix))
    eq_(model.linear1.in_features, embedding_dim)
    eq_(model.linear1.out_features,128)
    eq_(model.linear2.in_features, 128)
    eq_(model.linear2.out_features, len(word_to_ix))
    
#6.3
def test_cbow_forward():
    global model, X_tr, word_to_ix
    
    cv = cbow.build_context([X_tr[5]], context_size=2)
    ct = cbow.make_context_tensors(cv, word_to_ix, "cpu")
    
    probs = model(ct[0][0])
    
    assert_almost_equal(probs[0][0].item(),-9.4517211914, places=4)
    assert_almost_equal(probs[0][4].item(),-8.7454204559, places=4)