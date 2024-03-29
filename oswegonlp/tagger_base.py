from oswegonlp import scorer
from oswegonlp import preprocessing
from oswegonlp import classifier_base
from oswegonlp import bilstm
from oswegonlp.constants import DEV_FILE, OFFSET, TRAIN_FILE, UNK
import operator
from collections import defaultdict

argmax = lambda x : max(x.items(),key=lambda y : y[1])[0]

def make_classifier_tagger(weights):
    """
    :param weights: a defaultdict of classifier weights
    :returns: a function that takes a list of words, and a list of candidate tags, and returns tags for all words
    :rtype: function
    """
    tags_set = set()
    vocabs_set = set()
    for weight in weights:
        tags_set.add(weight[0])
        vocabs_set.add(weight[1])
    tags = list(tags_set)
    vocabs = list(vocabs_set)

    repeat_tag = argmax({tag: weights[(tag, OFFSET)] for tag in tags})
    classifier_map = defaultdict(lambda : repeat_tag)

    for word in vocabs:
        classifier_map[word] = argmax({tag: weights[(tag, word)] for tag in tags})

    def classify(words, all_tags):
        """This nested function should return a list of tags, computed using a classifier with the weights passed as arguments to make_classifier_tagger and using basefeatures for each token (just the token and the offset)
        :param words: list of words
        :param all_tags: all possible tags
        :returns: list of tags
        :rtype: list
        """
        return [classifier_map[word] for word in words]

    return classify


# compute tag with most unique word types: check if needed?
def most_unique_tag(weights, alltags):
    tag_uniq_counts = {tag: len([tup[0] for tup in weights.keys() if tup[0] == tag]) for tag in alltags}
    return argmax(tag_uniq_counts)


def apply_tagger(tagger, outfilename, all_tags=None, trainfile=TRAIN_FILE, testfile=DEV_FILE):
    if all_tags is None:
        all_tags = set()

        # this is slow
        for i, (words, tags) in enumerate(preprocessing.conll_seq_generator(trainfile)):
            for tag in tags:
                all_tags.add(tag)

    with open(outfilename, 'w') as outfile:
        for words, _ in preprocessing.conll_seq_generator(testfile):
            pred_tags = tagger(words, all_tags)
            for i, tag in enumerate(pred_tags):
                outfile.write(tag + '\n')
            outfile.write('\n')


def eval_tagger(tagger, outfilename, all_tags=None, trainfile=TRAIN_FILE, testfile=DEV_FILE):
    """Calculate confusion_matrix for a given tagger
    Parameters:
    tagger -- Function mapping (words, possible_tags) to an optimal
              sequence of tags for the words
    outfilename -- Filename to write tagger predictions to
    testfile -- (optional) Filename containing true labels
    Returns:
    confusion_matrix -- dict of occurences of (true_label, pred_label)
    """
    apply_tagger(tagger, outfilename, all_tags, trainfile, testfile)
    return scorer.get_confusion(testfile, outfilename)  # run the scorer on the prediction file


def apply_model(model, outfilename, word_to_ix, all_tags=None, trainfile=TRAIN_FILE, testfile=DEV_FILE):
    """
    applies the model on the data and writes the best sequence of tags to the outfile
    """
    if all_tags is None:
        all_tags = set()

        # this is slow
        for i, (words, tags) in enumerate(preprocessing.conll_seq_generator(trainfile)):
            for tag in tags:
                all_tags.add(tag)

    with open(outfilename, 'w') as outfile:
        for words, _ in preprocessing.conll_seq_generator(testfile):
            seq_words = bilstm.prepare_sequence(words, word_to_ix)
            pred_tags = model.predict(seq_words)
            for i, tag in enumerate(pred_tags):
                outfile.write(tag + '\n')
            outfile.write('\n')


def eval_model(model, outfilename, word_to_ix, all_tags=None, trainfile=TRAIN_FILE, testfile=DEV_FILE):
    """Calculate confusion_matrix for a given model
    Parameters:
    tagger -- Model mapping (words) to an optimal
              sequence of tags for the words
    outfilename -- Filename to write tagger predictions to
    testfile -- (optional) Filename containing true labels
    Returns:
    confusion_matrix -- dict of occurences of (true_label, pred_label)
    """
    apply_model(model, outfilename, word_to_ix, all_tags, trainfile, testfile)
    return scorer.get_confusion(testfile, outfilename)  # run the scorer on the prediction file