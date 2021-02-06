# Transformers Notebook 

# This notebook contains the implementation of our BERT and Google Universal Sentence Encoder (GUSE) transformers. 

import numpy as np
import pandas as pd

import torch
import transformers as ppb
import tensorflow as tf
import tensorflow_hub as hub

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB


# Google Universal Sentence Encoder (GUSE) Transformer

# install the following
# run 'pip install tensorflow==1.15'
# run 'pip install "tensorflow_hub>=0.6.0"'
# run 'pip3 install tensorflow_text==1.15'

#download the model to local so it can be used again and again
get_ipython().system('mkdir /content/module_useT')
# Download the module, and uncompress it to the destination folder. 
get_ipython().system('curl -L "https://tfhub.dev/google/universal-sentence-encoder-large/3?tf-hub-format=compressed" | tar -zxvC /content/module_useT')

embed = hub.Module("/content/module_useT")

# define the GUSE transformer
def encodeData(messages): 
  # Reduce logging output.
  tf.logging.set_verbosity(tf.logging.ERROR)
  with tf.Session() as session:
      session.run([tf.global_variables_initializer(), tf.tables_initializer()])
      message_embeddings = session.run(embed(messages))

  final_embeddings = pd.DataFrame(data=message_embeddings)

  return final_embeddings

# transform the text features to word embeddings using GUSE
training_regular = pd.read_csv('../data/training-set.csv')['selftext']
new_training_regular = encodeData(training_regular)
new_training_regular.to_csv('guse-training-features.csv')


# BERT Transformer
# install the following
# run'pip install transformers'

# importing bert-base, tokenizers, and models from libraries
model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

# defining the BERT transformer
# Look at the below comments to determine whether you want to output the 2-dimensional or 3-dimensional BERT features.
def getFeatures(batch_1):

  tokenized = batch_1[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True, truncation=True, max_length=512)))

  max_len = 0
  for i in tokenized.values:
      if len(i) > max_len:
          max_len = len(i)

  padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])


  attention_mask = np.where(padded != 0, 1, 0)
  attention_mask.shape


  input_ids = torch.tensor(padded)  
  attention_mask = torch.tensor(attention_mask)

  with torch.no_grad():
      last_hidden_states = model(input_ids, attention_mask=attention_mask)

  # features = last_hidden_states[0][:,0,:].numpy() # use this line if you want the 2D BERT features
  # features = last_hidden_states[0].numpy() # use this line if you want the 3D BERT features 

  return features

df = pd.read_csv('real-training-set.csv', delimiter=',')
df = df[['selftext', 'is_suicide']]
df = df.rename(columns={'selftext': 0, 'is_suicide': 1})

bert_features = getFeatures(df)
np.savetxt("bert-training-features.csv", bert_features, delimiter=',')