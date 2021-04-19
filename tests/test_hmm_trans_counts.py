from nose.tools import with_setup, eq_, assert_almost_equal

from oswegonlp.constants import START_TAG, END_TAG, TRAIN_FILE
from oswegonlp import hmm, most_common
import numpy as np

def setup():
    global tag_trans_counts, hmm_trans_weights
    tag_trans_counts = most_common.get_tag_trans_counts(TRAIN_FILE)
    

# 4.1a
def test_tag_trans_counts():
    global tag_trans_counts
    
    eq_(tag_trans_counts['DET']['NOUN'],9671)
    eq_(tag_trans_counts[START_TAG]['NOUN'],782)
    eq_(tag_trans_counts[START_TAG]['PUNCT'],433)