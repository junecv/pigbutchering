import streamlit as st

text = st.text_input('check if you\'re the \U0001F437', 'Hi Disky, how is your business doing?')

if text:
    out = 'business' in text
    st.write('text include keyword business', out)