import dashboard
import business
import streamlit as st

# rend l'affichage large et responsive
st.set_page_config(layout="wide")

PAGES = {
    "Scoring customer": dashboard,
    "Business value": business}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]

page.app()