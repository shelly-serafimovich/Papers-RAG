from upload_arcticles import find_similar_articles
from streamlit.components.v1 import html
import streamlit as st


def query_search_page():
    st.title("🔍 Query Search Page")

    # Input field for user query
    user_input = st.text_input("Enter your query:", placeholder="Type your search query here...")

    # Search button
    if st.button("Search"):
        if user_input:
            similar_articles = find_similar_articles(user_input)

            st.write("## Search Results:")
            for idx, article in enumerate(similar_articles, start=1):
                title = article['metadata']['title']
                abstract = article['metadata']['abstract']

                st.markdown(f"""
                    <div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                        <h3 style="color: #2c3e50;">{title}</h3>
                        <p style="color: #34495e;">{abstract}</p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Please enter a query to search.")
