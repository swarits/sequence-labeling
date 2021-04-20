from oswegonlp.constants import OFFSET
import numpy as np
import operator

# use this to find the highest-scoring label
def argmax(scores):
    items = list(scores.items())
    items.sort()
    return items[np.argmax([i[1] for i in items])][0]

def make_feature_vector(x,y):
    """take a counter of base features and a label; return a dict of features, corresponding to f(x,y)

    :param x: counter of base features
    :param y: label string
    :returns: dict of features, f(x,y)
    :rtype: dict

    """
    fv = {(y, OFFSET): 1}
    for feature, count in x.items():
        fv[(y, feature)] = count
    return fv
    #raise NotImplementedError
    
    
def compute_score(x,y,weights):
    total = 0
    for feature, count in make_feature_vector(x,y).items():
        if feature in weights:
            total += weights[feature] * count
    return total

def compute_scores(x,labels,weights):
    scores = {}
    for l in labels:
        scores[l] = compute_score(x,l,weights)
        #scores.append((l,))
    return scores

def predict(x,weights,labels):
    """prediction function

    :param base_features: a dictionary of base features and counts
    :param weights: a defaultdict of features and weights. features are tuples (label,base_feature).
    :param labels: a list of candidate labels
    :returns: top scoring label, scores of all labels
    :rtype: string, dict

    """
    scores = compute_scores(x, labels, weights)
    return argmax(scores),scores
    #raise NotImplementedError
