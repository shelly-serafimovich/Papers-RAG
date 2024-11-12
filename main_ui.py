import streamlit as st
from query_ui import query_search_page
from pdf_ui import pdf_search_page

# Set up session state to keep track of the current page
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "Query Search"

# Sidebar for navigation with button-like options
st.sidebar.title("Navigation")
if st.sidebar.button("🔍 Query Search"):
    st.session_state['current_page'] = "Query Search"
if st.sidebar.button("📄 PDF Search"):
    st.session_state['current_page'] = "PDF Search"

# Load the selected page
if st.session_state['current_page'] == "Query Search":
    query_search_page()
elif st.session_state['current_page'] == "PDF Search":
    pdf_search_page()
