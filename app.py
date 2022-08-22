import streamlit as st
import pickle as pkl
import tensorflow as tf
from tensorflow.keras import layers

# retrieve text vectorization layer 
tv_spec = pkl.load(Open('saved_model/tv_layer.pkl', 'rb'))
text_vectorizer = layers.TextVectorization.from_config(tv_spec['config'])
text_vectorizer.set_weights(tv_spec['weights'])

# load model
model = tf.keras.models.load_model('saved_model/dnn_model')

# get input
text = st.text_input('check if you\'re the \U0001F437', 'Hi Disky, how is your business doing?')

if text:
    text_vector = text_vectorizer(text)
    out = model.predict(text_vector)[0][0]
    st.write('Your \U0001F437 score is', out)
    if out > 0.5:
        st.write('I smell \U0001F953')
    else: st.write('Well, think twice.')