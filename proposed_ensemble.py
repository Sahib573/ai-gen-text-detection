import numpy as np
import maths
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split

# Load trained models from h5 files
bert_model = load_model('../models/Debert.keras')
deberta_model = load_model('../models/model-bert.keras')
ensemble_model = load_model('../models/Voting Classifier Model.h5')
Train_Data= pd.read_csv("../dataset/train.csv")
x_train,x_test,y_train,y_test=train_test_split(Train_Data.text,Train_Data.generated,test_size=0.009,shuffle=True)
# Define initial weights
initial_bert_weight = 0.33
initial_deberta_weight = 0.34
initial_ensemble_weight = 0.33

# Define validation accuracy for each model (replace these with actual validation accuracy values)
bert_val_accuracy = 0.85
deberta_val_accuracy = 0.82
ensemble_val_accuracy = 0.88

# Define number of iterations for fine-tuning
num_iterations = 10

# Define a function to dynamically adjust learning rate during training
def lr_scheduler(batch_size=8, mode='cos', epochs=10, plot=False):
    lr_start, lr_max, lr_min = 0.6e-6, 0.5e-6 * batch_size, 0.3e-6
    lr_ramp_ep, lr_sus_ep, lr_decay = 1, 0, 0.75

    def lrfn(epoch):  # Learning rate update function
        decay_total_epochs, decay_epoch_index = epochs - lr_ramp_ep - lr_sus_ep + 3, epoch - lr_ramp_ep - lr_sus_ep
        phase = math.pi * decay_epoch_index / decay_total_epochs
        lr = (lr_max - lr_min) * 0.5 * ( 1 + math.cos(phase)) + lr_min
        return lr
# Create a learning rate scheduler
scheduler = LearningRateScheduler(lr_scheduler)

# Define optimizer with initial learning rate
optimizer = Adam(lr=0.01)

# Compile models with optimizer
bert_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
deberta_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
ensemble_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Fine-tune weights
for i in range(num_iterations):
    # Train models with current weights
    # Note: Replace X_train, y_train with your training data
    bert_model.fit(x_train, y_train, epochs=1, callbacks=[scheduler])
    deberta_model.fit(x_train, y_train, epochs=1, callbacks=[scheduler])
    ensemble_model.fit(x_train, y_train, epochs=1, callbacks=[scheduler])

    # Calculate total validation accuracy
    total_accuracy = bert_val_accuracy * initial_bert_weight + \
                     deberta_val_accuracy * initial_deberta_weight + \
                     ensemble_val_accuracy * initial_ensemble_weight
    
    # Update weights based on individual model performance
    bert_weight = (bert_val_accuracy * initial_bert_weight) / total_accuracy
    deberta_weight = (deberta_val_accuracy * initial_deberta_weight) / total_accuracy
    ensemble_weight = (ensemble_val_accuracy * initial_ensemble_weight) / total_accuracy
    
    # Update initial weights with learning rate
    initial_bert_weight += optimizer.lr.numpy() * (bert_weight - initial_bert_weight)
    initial_deberta_weight += optimizer.lr.numpy() * (deberta_weight - initial_deberta_weight)
    initial_ensemble_weight += optimizer.lr.numpy() * (ensemble_weight - initial_ensemble_weight)

    # Normalize weights
    total_weight = initial_bert_weight + initial_deberta_weight + initial_ensemble_weight
    initial_bert_weight /= total_weight
    initial_deberta_weight /= total_weight
    initial_ensemble_weight /= total_weight

    # Print fine-tuned weights
    print("Fine-Tuned BERT Weight:", initial_bert_weight)
    print("Fine-Tuned DeBERTa Weight:", initial_deberta_weight)
    print("Fine-Tuned Ensemble Weight:", initial_ensemble_weight)