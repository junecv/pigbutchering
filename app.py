import streamlit as st
import pickle as pkl
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# retrieve text vectorization layer 
tv_spec = pkl.load(open('model/tv_layer.pkl', 'rb'))
text_vectorizer = layers.TextVectorization.from_config(tv_spec['config'])
text_vectorizer.set_weights(tv_spec['weights'])

# function to create model Simple DNN
def get_model(hidden_dim = 8):
  inputs = keras.Input(shape=(35,), dtype = "int64")
  x = layers.Dense(hidden_dim, activation = "relu")(inputs)
  outputs = layers.Dense(1, activation = 'sigmoid')(x)
  model = keras.Model(inputs, outputs)
  model.compile(
    loss = "binary_crossentropy",
    optimizer = keras.optimizers.Adam(learning_rate = 0.01),
    metrics = ['accuracy']
    )
  return model

model = get_model()
model.load_weights('model/dnn_model.h5')

# get input
text = st.text_input('check if you\'re the \U0001F437', 'Hi Disky, how is your business doing?')

if text:
    text_vector = text_vectorizer(text)
    out = model.predict(text_vector)
    st.json(out)
    # st.write('Your \U0001F437 score is', out)
    # if out > 0.5:
    #     st.write('I smell \U0001F953')
    # else: st.write('Well, think twice.')