import streamlit as st


# Function to extract text from PDF
def extract_text_from_pdf(uploaded_pdf):
    text = "hello world"
    return text


# Function to simulate finding similar articles (replace this with your actual function)
def find_similar_articles_from_text(text):
    return [
        {
            "metadata": {
                "title": "PDF-Based Machine Learning Insights",
                "abstract": "This article elaborates on how PDFs can be parsed to extract valuable content for learning models."
            }
        },
        {
            "metadata": {
                "title": "Innovative PDF Content Analysis",
                "abstract": "Exploring methodologies for analyzing PDFs and leveraging them in AI pipelines."
            }
        },
    ]


def pdf_search_page():
    st.title("📄 PDF Search Page")

    # File uploader for PDF
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    # Search button
    if st.button("Search") and uploaded_file is not None:
        with st.spinner("Extracting text from PDF..."):
            extracted_text = extract_text_from_pdf(uploaded_file)

        if extracted_text:
            similar_articles = find_similar_articles_from_text(extracted_text)

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
            st.error("No text could be extracted from the PDF.")
    elif not uploaded_file:
        st.info("Please upload a PDF file to proceed.")
