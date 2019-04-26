# first run a few imports:
import tensorflow as tf
import numpy as np
from pprint import pprint
import pickle

# the paths to our training and validation data, English side
train_e_path = 'data/training/hansards.36.2.e.gz'
train_f_path = 'data/training/hansards.36.2.f.gz'
dev_e_path = 'data/validation/dev.e.gz'
dev_f_path = 'data/validation/dev.f.gz'
dev_wa = 'data/validation/dev.wa.nonullalign'

# check utils.py if you want to see how smart_reader and bitext_reader work in detail
from utils import smart_reader, bitext_reader

    
def bitext_reader_demo(src_path, trg_path):
  """Demo of the bitext reader."""
 
  # create a reader
  src_reader = smart_reader(src_path)
  trg_reader = smart_reader(trg_path)
  bitext = bitext_reader(src_reader, trg_reader)

  # to see that it really works, try this:
  print(next(bitext))
  print(next(bitext))
  print(next(bitext))
  print(next(bitext))  


bitext_reader_demo(train_e_path, train_f_path)


# To see how many sentences are left if you filter by length, you can do this:

def demo_number_filtered_sentence_pairs(src_path, trg_path):
  src_reader = smart_reader(src_path)
  trg_reader = smart_reader(trg_path)
  max_length = 30
  bitext = bitext_reader(src_reader, trg_reader, max_length=max_length)   
  num_sentences = sum([1 for _ in bitext])
  print("There are {} sentences with max_length = {}".format(num_sentences, max_length))
  
  
demo_number_filtered_sentence_pairs(train_e_path, train_f_path)

# check vocabulary.py to see how the Vocabulary class is defined
from vocabulary import OrderedCounter, Vocabulary 


def vocabulary_demo():

  # We used up a few lines in the previous example, so we set up
  # our data generator again.
  corpus = smart_reader(train_e_path)    

  # Let's create a vocabulary given our (tokenized) corpus
  vocabulary = Vocabulary(corpus=corpus)
  print("Original vocabulary size: {}".format(len(vocabulary)))

  # Now we only keep the highest-frequency words
  vocabulary_size=1000
  vocabulary.trim(vocabulary_size)
  print("Trimmed vocabulary size: {}".format(len(vocabulary)))

  # Now we can get word indexes using v.get_word_id():
  for t in ["<PAD>", "<UNK>", "the"]:
    print("The index of \"{}\" is: {}".format(t, vocabulary.get_token_id(t)))

  # And the inverse too, using v.i2t:
  for i in range(10):
    print("The token with index {} is: {}".format(i, vocabulary.get_token(i)))

  # Now let's try to get a word ID for a word not in the vocabulary
  # we should get 1 (so, <UNK>)
  for t in ["!@!_not_in_vocab_!@!"]:
    print("The index of \"{}\" is: {}".format(t, vocabulary.get_token_id(t)))
    
    
vocabulary_demo()


# Using only 1000 words will result in many UNKs, but
# it will make training a lot faster. 
# If you have a fast computer, a GPU, or a lot of time,
# try with 10000 instead.
max_tokens=1000

corpus_e = smart_reader(train_e_path)    
vocabulary_e = Vocabulary(corpus=corpus_e, max_tokens=max_tokens)
pickle.dump(vocabulary_e, open("vocabulary_e.pkl", mode="wb"))
print("English vocabulary size: {}".format(len(vocabulary_e)))

corpus_f = smart_reader(train_f_path)    
vocabulary_f = Vocabulary(corpus=corpus_f, max_tokens=max_tokens)
pickle.dump(vocabulary_f, open("vocabulary_f.pkl", mode="wb"))
print("French vocabulary size: {}".format(len(vocabulary_f)))
print()


def sample_words(vocabulary, n=5):
  """Print a few words from the vocabulary."""
  for _ in range(n):
    token_id = np.random.randint(0, len(vocabulary) - 1)
    print(vocabulary.get_token(token_id))


print("A few English words:")
sample_words(vocabulary_e, n=5)
print()

print("A few French words:")
sample_words(vocabulary_f, n=5)



from utils import iterate_minibatches, prepare_data


src_reader = smart_reader(train_e_path)
trg_reader = smart_reader(train_f_path)
bitext = bitext_reader(src_reader, trg_reader)


for batch_id, batch in enumerate(iterate_minibatches(bitext, batch_size=4)):

  print("This is the batch of data that we will train on, as tokens:")
  pprint(batch)
  print()

  x, y = prepare_data(batch, vocabulary_e, vocabulary_f)

  print("These are our inputs (i.e. words replaced by IDs):")
  print(x)
  print()
  
  print("These are the outputs (the foreign sentences):")
  print(y)
  print()

  if batch_id > 0:
    break  # stop after the first batch, this is just a demonstration


# check neuralibm1.py for the Model code
# Implement Neural IBM 1 on this class
from neuralibm1 import NeuralIBM1Model

# check neuralibm1trainer.py for the Trainer code
from neuralibm1trainer import NeuralIBM1Trainer

print(len(vocabulary_e))


tf.reset_default_graph()
with tf.Session() as sess:

  # some hyper-parameters
  # tweak them as you wish
  batch_size=16  # on CPU, use something much smaller e.g. 1-16
  max_length=30
  lr = 0.001
  lr_decay = 0.0  # set to 0.0 when using Adam optimizer (default)
  emb_dim = 64
  mlp_dim = 128
  
  # our model
  model = NeuralIBM1Model(
    x_vocabulary=vocabulary_e, y_vocabulary=vocabulary_f, 
    batch_size=batch_size, emb_dim=emb_dim, mlp_dim=mlp_dim, session=sess)
  
  # our trainer
  trainer = NeuralIBM1Trainer(
    model, train_e_path, train_f_path, 
    dev_e_path, dev_f_path, dev_wa,
    num_epochs=10, batch_size=batch_size, 
    max_length=max_length, lr=lr, lr_decay=lr_decay, session=sess)

  # now first TF needs to initialize all the variables
  print("Initializing variables..")
  sess.run(tf.global_variables_initializer())

  # now we can start training!
  print("Training started..")
  trainer.train()


