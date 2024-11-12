import re
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import requests
import fitz  # PyMuPDF for PDF extraction
from xml.etree import ElementTree as ET
import pytesseract
from PIL import Image

# Initialize the SentenceTransformer model and Pinecone client
model = SentenceTransformer("all-MiniLM-L6-v2")
pc = Pinecone(api_key="65adfe61-8c99-4c68-951e-e2d42e7884df")
index = pc.Index("document-embeddings")


# Function to extract the article ID from the URL
def extract_arxiv_id(url):
    match = re.search(r'arxiv.org\/abs\/(\d+\.\d+)', url)
    return match.group(1) if match else None


# Function to fetch title and abstract from arXiv using the ID
def fetch_arxiv_metadata(arxiv_id):
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    response = requests.get(url)
    if response.status_code == 200:
        root = ET.fromstring(response.text)
        entry = root.find('{http://www.w3.org/2005/Atom}entry')
        if entry is not None:
            title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
            abstract = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip()
            return title, abstract
    return None, None


# Function to extract arXiv ID from an image using OCR
def extract_arxiv_id_from_image(image_path):
    try:
        image = Image.open(image_path).convert('L')
        ocr_result = pytesseract.image_to_string(image)
        arxiv_id_match = re.search(r'\b(\d{4}\.\d{4,5})(v\d+)?\b', ocr_result)
        return arxiv_id_match.group(1) if arxiv_id_match else None
    except Exception as e:
        print(f"Error during OCR processing: {e}")
        return None


# Function to extract text from the first page of a PDF using OCR
def extract_text_from_pdf_with_ocr(pdf_path):
    try:
        pdf_document = fitz.open(pdf_path)
        if len(pdf_document) > 0:
            page = pdf_document.load_page(0)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img)
            return text
        else:
            print("PDF has no pages.")
            return None
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None


# Function to encode text using SentenceTransformer
def encode_text(text):
    return model.encode(text)


# Function to get top N similar articles from Pinecone
def get_similar_articles_from_pinecone(target_vector, arxiv_id=None, top_n=5):
    query_result = index.query(vector=target_vector.tolist(), top_k=top_n + 1, include_metadata=True)
    filtered_matches = [
        match for match in query_result['matches'] if not arxiv_id or match['id'] != arxiv_id
    ]
    return filtered_matches[:top_n]


# Main function to handle user input and find similar articles
def find_similar_articles(user_input):
    arxiv_id = None

    if user_input.endswith('.pdf'):
        text = extract_text_from_pdf_with_ocr(user_input)
        if text:
            arxiv_id_match = re.search(r'\b(\d{4}\.\d{4,5})(v\d+)?\b', text)
            arxiv_id = arxiv_id_match.group(1) if arxiv_id_match else None
    elif 'arxiv.org' in user_input:
        arxiv_id = extract_arxiv_id(user_input)
    elif user_input.endswith(('.png', '.jpg', '.jpeg')):
        arxiv_id = extract_arxiv_id_from_image(user_input)
    else:
        arxiv_id = user_input

    if not arxiv_id:
        print("No valid arXiv ID found.")
        return

    query_result = False  # Placeholder for actual query existence check
    if query_result:
        target_vector = query_result.matches[0].values
    else:
        title, abstract = fetch_arxiv_metadata(arxiv_id)
        if not title or not abstract:
            print("Could not retrieve metadata from arXiv.")
            return
        combined_text = title + ' ' + abstract
        target_vector = encode_text(combined_text)

    similar_articles = get_similar_articles_from_pinecone(target_vector, arxiv_id=arxiv_id)
    print("Similar Articles:")
    for match in similar_articles:
        print(f"Title: {match['metadata']['title']}, Similarity Score: {match['score']}")

    return similar_articles


# Example usages
if __name__ == "__main__":
    user_input = "0803.1752"
    similar_articles = find_similar_articles(user_input)
    print(similar_articles[0])

