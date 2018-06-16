from keras.models import Sequential
import numpy as np
from keras.datasets import imdb
import pickle


"""
IMDB dataset: a set of 50,000 highly polarized reviews from the
Internet Movie Database. They’re split into 25,000 reviews for training and 25,000
reviews for testing, each set consisting of 50% negative and 50% positive reviews.

The argument num_words=10000 means you’ll only keep the top 10,000 most frequently
occurring words in the training data. Rare words will be discarded. This allows
you to work with vector data of manageable size.
"""

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

imdb_dict = {}
imdb_dict['x_train'] = vectorize_sequences(train_data)
imdb_dict['x_test'] = vectorize_sequences(test_data)

imdb_dict['y_train'] = np.asarray(train_labels).astype('float32')
imdb_dict['y_test'] = np.asarray(test_labels).astype('float32')

imdb_dict['x_val'] = imdb_dict['x_train'][:10000]
imdb_dict['y_val'] = imdb_dict['y_train'][:10000]
imdb_dict['x_p_train_1'] = imdb_dict['x_train'][10000:20000]
imdb_dict['y_p_train_1'] = imdb_dict['y_train'][10000:20000]
imdb_dict['x_p_train_2'] = imdb_dict['x_train'][20000:25000]
imdb_dict['y_p_train_2'] = imdb_dict['y_train'][20000:25000]
imdb_dict['x_p_train'] = imdb_dict['x_train'][10000:25000]
imdb_dict['y_p_train'] = imdb_dict['y_train'][10000:25000]


pickle.dump( imdb_dict, open( "imdb.p", "wb" ) )


"""


imdb_dict_causal = {}
imdb_dict_causal['x_train'] = vectorize_sequences(train_data)
imdb_dict_causal['x_test'] = vectorize_sequences(test_data)

imdb_dict_causal['y_train'] = np.asarray(train_labels).astype('float32')
imdb_dict_causal['y_test'] = np.asarray(test_labels).astype('float32')


imdb_dict_causal['x_val'] = imdb_dict_causal['x_train'][:10000]
imdb_dict_causal['y_val'] = imdb_dict_causal['y_train'][:10000]
imdb_dict_causal['partial_x_train_1'] = imdb_dict_causal['x_train'][10000:20000]
imdb_dict_causal['partial_y_train_1'] = imdb_dict_causal['y_train'][10000:20000 ]
imdb_dict_causal['partial_x_train_2'] = imdb_dict_causal['x_train'][20000:25000 ]
imdb_dict_causal['partial_y_train_2'] = imdb_dict_causal['y_train'][20000:25000 ]
imdb_dict_causal['partial_x_train_3'] = imdb_dict_causal['x_train'][20000:25000 ]
imdb_dict_causal['partial_y_train_3'] = imdb_dict_causal['y_train'][20000:25000 ]
pickle.dump( imdb_dict_causal, open( "imdb_causal.p", "wb" ) )
"""

