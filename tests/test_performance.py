from nose.tools import with_setup, ok_, eq_, assert_almost_equal, nottest, assert_not_equal
import torch
from oswegonlp.constants import * 
import numpy as np

#7.1a
def test_model_en_dev_accuracy1():
    confusion = scorer.get_confusion(DEV_FILE,'model-dev-en.preds')
    acc = scorer.accuracy(confusion)
    ok_(acc > .905)

#7.1b
def test_model_en_dev_accuracy2():
    confusion = scorer.get_confusion(DEV_FILE,'model-dev-en.preds')
    acc = scorer.accuracy(confusion)
    ok_(acc > .91)
    
#7.1c  
def test_model_en_test_accuracy1():
    confusion = scorer.get_confusion(TEST_FILE,'model-te-en.preds')
    acc = scorer.accuracy(confusion)
    ok_(acc > .905)
    
#7.1d    
def test_model_en_test_accuracy2():
    confusion = scorer.get_confusion(TEST_FILE,'model-te-en.preds')
    acc = scorer.accuracy(confusion)
    ok_(acc > .91)

#7.1e
def test_model_nr_dev_accuracy1():
    confusion = scorer.get_confusion(NR_DEV_FILE,'model-dev-nr.preds')
    acc = scorer.accuracy(confusion)
    ok_(acc > .905)
    
#7.1f
def test_model_nr_dev_accuracy2():
    confusion = scorer.get_confusion(NR_DEV_FILE,'model-dev-nr.preds')
    acc = scorer.accuracy(confusion)
    ok_(acc > .91)
    
#7.1g   
def test_model_nr_test_accuracy1():
    confusion = scorer.get_confusion(NR_TEST_FILE,'model-te-nr.preds')
    acc = scorer.accuracy(confusion)
    ok_(acc > .905)

#7.1h   
def test_model_nr_test_accuracy2():
    confusion = scorer.get_confusion(NR_TEST_FILE,'model-te-nr.preds')
    acc = scorer.accuracy(confusion)
    ok_(acc > .91)






