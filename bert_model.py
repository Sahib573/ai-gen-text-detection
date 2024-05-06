# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import tensorflow_text as text  # Registers the ops.
import tensorflow_hub as hub
import matplotlib.pyplot as plt
train_essay = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/train_essays.csv")
train_essay
test_essay = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/test_essays.csv")
test_essay
sample_sub = pd.read_csv("//kaggle/input/llm-detect-ai-generated-text/test_essays.csv")
sample_sub
train_essay.info()
train_essay['prompt_id'].value_counts()
sns.countplot(x=train_essay['prompt_id'])
plt.show()
train_essay['prompt_id'].value_counts().plot(kind="pie",autopct="%.1f%%")
plt.title("Prompt ID")
plt.show()
train_essay['generated'].value_counts().plot(kind="pie",autopct="%.1f%%") # to see what to take as base
plt.title("Target label")
plt.show()
train_essay.head()
ai_df = train_essay[train_essay['generated']==1]
ai_df
train_essay
stopwords_text = """"i me my myself we our ours ourselves you you're you've you'll you'd your yours yourself yourselves he him his himself she she's her hers herself it it's its itself they them their theirs themselves what which who whom this that that'll these those am is are was were be been being have has had having do does did doing a an the and but if or because as until while of at by for with about against between into through during before after above below to from up down in out on off over under again further then once here there when where why how all any both each few more most other some such only own same so than too very s t can will just don don't should should've now d ll m o re ve y ain aren aren't couldn couldn't didn didn't doesn doesn't hadn hadn't hasn hasn't haven haven't isn isn't ma mightn mightn't mustn mustn't needn needn't shan shan't shouldn shouldn't wasn wasn't weren weren't won won't wouldn wouldn't"
"""
stopwords_list = stopwords_text.split()
len(stopwords_list)
df = pd.read_csv("/kaggle/input/dataset-4/Training_Essay_Data.csv")
df
df2 = pd.read_csv("/kaggle/input/daigt-proper-train-dataset/train_drcat_04.csv")
df2 = df2[['text','label']]
df2.columns = ['text','generated']
df2
df3 = pd.read_csv("/kaggle/input/llm-7-prompt-training-dataset/train_essays_RDizzl3_seven_v1.csv")
df3.columns = ['text','generated']
df3
train_data = pd.concat([df3,df2,df],axis=0,ignore_index=True)
train_data
train_data.drop_duplicates(inplace=True,ignore_index=True)
train_data
# Dataset
d1 = pd.read_csv("/kaggle/input/daigt-data-llama-70b-and-falcon180b/falcon_180b_v1.csv")
d1
# Dataset
d2 = pd.read_csv("/kaggle/input/daigt-data-llama-70b-and-falcon180b/llama_70b_v1.csv")
d2

data = pd.concat([d1,d2],axis=0,ignore_index=True)
data['generated'] = 1
data.columns = ['text','writing_prompt','generated']
data = data[['text','generated']]
data
Train_Data = pd.concat([train_data,data],axis=0,ignore_index=True)
Train_Data
d = pd.read_csv("/kaggle/input/llm-generated-essay-using-palm-from-google-gen-ai/LLM_generated_essay_PaLM.csv")
d
d['generated'] = d['generated'].astype(int)
dd = d[['text','generated']]
dd
Train_Data  =pd.concat([Train_Data,dd],axis=0,ignore_index=True)
Train_Data
Train_Data.drop_duplicates(inplace=True,ignore_index=True)
Train_Data
Train_Data['generated'].value_counts()

Train_Data.generated.value_counts().plot(kind='pie',autopct="%.1f%%")
plt.title("Target Column Distributions")
plt.show()

sns.countplot(x=Train_Data['generated'])
plt.show
Train_Data.to_csv("train.csv")
x_train,x_test,y_train,y_test=train_test_split(Train_Data.text,Train_Data.generated,test_size=0.009,shuffle=True)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
x_train
y_train
model_path ="/kaggle/input/bert/tensorflow2/bert-en-uncased-l-12-h-768-a-12/2"
preprocess_path = "/kaggle/input/bert/tensorflow2/en-uncased-preprocess/3/"

text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
preprocessor = hub.KerasLayer(preprocess_path)
encoder_inputs  = preprocessor(text_input) # this is basically the preprocessed text

## Use BERT Model
encoder = hub.KerasLayer(model_path,trainable=True)
outputs = encoder(encoder_inputs)
pooled_output = outputs['pooled_output'] # [batch_size, 512].
sequence_output = outputs["sequence_output"] # [batch_size, seq_length, 512].
dropout = tf.keras.layers.Dropout(0.51 , name="dropout1")(pooled_output)
dense_2 = tf.keras.layers.Dense(64 , activation='relu')(dropout)
dropout = tf.keras.layers.Dropout(0.3 , name="dropout2")(dense_2)

dense_out = tf.keras.layers.Dense(1 , activation='sigmoid', name='output')(dropout)

model = tf.keras.Model(inputs=text_input, outputs=dense_out)
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
              loss='binary_crossentropy',
              metrics=["acc"])

checkpoint_filepath = 'checkpoint.hdf5'
metric = 'val_accuracy'
callback_list = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, monitor=metric,
                    verbose=2, save_best_only=True, mode='max')
history = model.fit(x_train, y_train , batch_size=8, callbacks=[callback_list],epochs=1 , validation_data=(x_test, y_test))
loss , acc = model.evaluate(x_train, y_train)
print("Accuracy on Train data:",acc)
loss , acc = model.evaluate(x_test, y_test)
print("Accuracy on Test data:",acc)

model.summary()
model.save("model-bert.h5")

from sklearn.metrics import confusion_matrix
import numpy as np
predictions = model.predict(x_train)
threshold = 0.4
binary_predictions = np.where(predictions < threshold, 0, 1)
binary_predictions= binary_predictions.flatten()
true_values = y_train.values.flatten()  
y_train_values
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(true_values, binary_predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
accuracy = np.trace(cm) / np.sum(cm)
print("Accuracy:", accuracy)


from sklearn.metrics import precision_score, recall_score, f1_score
precision = precision_score(true_values, binary_predictions)
recall = recall_score(true_values, binary_predictions)
f1 = f1_score(true_values, binary_predictions)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


from sklearn.metrics import confusion_matrix
import numpy as np

predictions = model.predict(x_test)
threshold = 0.3
binary_predictions = np.where(predictions < threshold, 0, 1)

binary_predictions_test= binary_predictions.flatten()
true_values = y_test.values.flatten()  

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(true_values, binary_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

accuracy = np.trace(cm) / np.sum(cm)
print("Accuracy:", accuracy)