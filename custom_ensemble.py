import sys
import gc

import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier

# Model Evaluation and Metrics
from sklearn.metrics import roc_auc_score

# Text Processing and Feature Extraction
from sklearn.feature_extraction.text import TfidfVectorizer

# Tokenizer and Preprocessing for NLP
from transformers import PreTrainedTokenizerFast
from tokenizers import (decoders,
                        models,
                        normalizers,
                        pre_tokenizers,
                        processors,
                        trainers,
                        Tokenizer)

# Dataset Handling and Progress Bar
from datasets import Dataset
from tqdm.auto import tqdm
test = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/test_essays.csv')
sub = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/sample_submission.csv')
org_train = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/train_essays.csv')

train1 = pd.read_csv("/kaggle/input/daigt-v2-train-dataset/train_v2_drcat_02.csv", sep=',')
train2 = pd.read_csv('/kaggle/input/daigt-proper-train-dataset/train_drcat_04.csv')

org_train = org_train.rename(columns={'generated': 'label'})

excluded_prompt_name_list = ['Distance learning','Grades for extracurricular activities','Summer projects']
train1 = train1[~(train1['prompt_name'].isin(excluded_prompt_name_list))]

# Concatenate them
train = pd.concat([org_train, train1, train2])
train = pd.read_csv("/kaggle/input/daigt-v2-train-dataset/train_v2_drcat_02.csv", sep=',')

# Reset index if necessary
train.reset_index(drop=True, inplace=True)
train = train.drop_duplicates(subset=['text'])
train.reset_index(drop=True, inplace=True)
train.head(2)
train.size
import string

# Tokenize and normalize the text
unique_words = set()
for text in train['text']:
    words = text.lower().split()  # Convert to lowercase and split into words
    unique_words.update(words)

# Remove punctuation from each word
unique_words = {word.strip(string.punctuation) for word in unique_words}

# Now, unique_words set contains all unique words
total_unique_words = len(unique_words)
print("Total unique words:", total_unique_words)
LOWERCASE = False
VOCAB_SIZE = 14000000
raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
raw_tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()] + [normalizers.Lowercase()] if LOWERCASE else [])
raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens)
dataset = Dataset.from_pandas(test[['text']])

def train_corp_iter(): 
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]
raw_tokenizer.train_from_iterator(train_corp_iter(), trainer=trainer)

# Wrapping the trained tokenizer with Huggingface's PreTrainedTokenizerFast for additional functionalities.
# This step integrates the tokenizer with Huggingface's ecosystem, enabling easy use with their models.
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=raw_tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]")

# Tokenizing the text data in the 'test' DataFrame and storing the results.
tokenized_texts_test = []
for text in tqdm(test['text'].tolist()):
    tokenized_texts_test.append(tokenizer.tokenize(text))

# Tokenizing the text data in the 'train' DataFrame and storing the results.
tokenized_texts_train = []
for text in tqdm(train['text'].tolist()):
    tokenized_texts_train.append(tokenizer.tokenize(text))

tokenized_texts_test[1]

def dummy(text):
    """
    A dummy function to use as tokenizer for TfidfVectorizer. It returns the text as it is since we already tokenized it.
    """
    return text

vectorizer = TfidfVectorizer(ngram_range=(3, 5), 
                             lowercase=False, 
                             sublinear_tf=True, 
                             analyzer = 'word',
                             tokenizer = dummy,
                             preprocessor = dummy,
                             token_pattern = None,
                             strip_accents='unicode')

# Fit vectorizer on test data to learn vocabulary
vectorizer.fit(tokenized_texts_test)

vocab = vectorizer.vocabulary_  # Extract learned vocabulary

# Reinitialize TfidfVectorizer for training set using test set's vocabulary
vectorizer = TfidfVectorizer(ngram_range=(3, 5), 
                             lowercase=False, 
                             sublinear_tf=True, 
                             vocabulary=vocab,
                             analyzer = 'word',
                             tokenizer = dummy,
                             preprocessor = dummy,
                             token_pattern = None, 
                             strip_accents='unicode')

# Transform training and test data into TF-IDF vectors
tf_train = vectorizer.fit_transform(tokenized_texts_train)
tf_test = vectorizer.transform(tokenized_texts_test)

# Cleanup: Free up memory
del vectorizer
gc.collect()
print_bool = False

if print_bool: 
    tf_demonstration_vector = tf_test.copy()
    tf_idf_array = tf_demonstration_vector.toarray()

    print("As can be seen, we do indeed have a sparse matrix:")
    print(type(tf_demonstration_vector), tf_demonstration_vector.shape)
    print("")
    print(tf_idf_array)

y_train = train['label'].values

def get_model():
    from catboost import CatBoostClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    clf = MultinomialNB(alpha=0.0225)
    sgd_model = SGDClassifier(max_iter=9000, tol=1e-4, loss="modified_huber", random_state=6743)
    p6={'n_iter': 3000,'verbose': -1,'objective': 'cross_entropy','metric': 'auc',
        'learning_rate': 0.00281909898961407, 'colsample_bytree': 0.78,
        'colsample_bynode': 0.8,
       }
    p6["random_state"] = 6743
    lgb=LGBMClassifier(**p6)
    cat=CatBoostClassifier(iterations=3000,
                           verbose=0,
                           random_seed=6543,
#                            l2_leaf_reg=6.6591278779517808,
                           learning_rate=0.002599066836106983,
                           subsample = 0.45,
                           allow_const_label=True,loss_function = 'CrossEntropy')
    
    rf_model = RandomForestClassifier(random_state=6743)
    
    weights = [0.1, 0.51, 0.28, 0.85, 0.35]
    
    ensemble = VotingClassifier(estimators=[('mnb',clf),
                                        ('sgd', sgd_model),
                                        ('lgb',lgb), 
                                        ('cat', cat),
                                        ('rf', rf_model)
                                        ],
                                weights=weights, voting='soft', n_jobs=-1)
    return ensemble

model = get_model()
print(model)
model.fit(tf_train, y_train)