import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import openai
import cohere
import genai

# Initialize SentenceTransformer model for embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Cohere client (replace 'your-api-key' with an actual API key)
co = cohere.Client('your-api-key')

# Initialize GenAI model
genai_client = genai.GenerativeModel("gemini-1.5-flash")


# Function to encode text using SentenceTransformer
def encode_text(text):
    embeddings = model.encode(text)
    return embeddings


# Function to call OpenAI GPT API
def call_gpt_api(query, abstract):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a relevance evaluator for scientific articles. Reply with only a "
                                          "single number between 1 and 10."},
            {"role": "user", "content": f"Question: {query}\nArticle Abstract: {abstract}\nRate the relevance from 1 "
                                        f"to 10. Only provide the number as an answer."}
        ]
    )
    score_text = response['choices'][0]['message']['content']
    return float(score_text.strip())


# Function to call Cohere API
def call_cohere_api(query, abstract):
    response = co.generate(
        model="command-xlarge-nightly",
        prompt=f"You are a relevance evaluator for scientific articles. Question: {query}\nArticle Abstract: {abstract}\nRate the relevance from 1 to 10. Only provide the number as an answer.",
        max_tokens=10
    )
    return float(response.generations[0].text.strip())


# Function to call Gemini API
def call_gemini_api(query, abstract):
    prompt_text = f"You are a relevance evaluator for scientific articles. Question: {query}\nArticle Abstract: {abstract}\nRate the relevance from 1 to 10. Only provide the number as an answer."
    response = genai_client.generate_content(prompt_text)
    return float(response.text)


# Function to retrieve articles from Pinecone
def retrieve_articles(query, top_k=5):
    query_embedding = encode_text(query)
    response = index.query(vector=query_embedding.tolist(), top_k=top_k, include_metadata=True, include_values=True)
    articles = [(match['metadata']['title'], match['metadata']['abstract']) for match in response['matches']]
    return articles


# Function to evaluate articles using GPT, Cohere, and Gemini
def evaluate_articles(query, articles):
    gpt_scores, cohere_scores, gemini_scores = [], [], []
    for title, abstract in articles:
        combined_text = f"Title: {title}\nAbstract: {abstract}"

        # OpenAI GPT Score
        gpt_score = call_gpt_api(query, combined_text)
        gpt_scores.append(gpt_score)

        # Cohere Score
        cohere_score = call_cohere_api(query, combined_text)
        cohere_scores.append(cohere_score)

        # Gemini Score
        gemini_score = call_gemini_api(query, combined_text)
        gemini_scores.append(gemini_score)

    avg_scores = [(g + c + a) / 3 for g, c, a in zip(gpt_scores, cohere_scores, gemini_scores)]
    return avg_scores


# Function to expand the query using GPT
def expand_query(query):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert in data science and information retrieval."},
            {"role": "user",
             "content": f"Provide additional keywords or phrases that could be incorporated into the following query to refine or broaden it, but do not answer the question. Return a concise list of relevant terms or phrases only. Query: '{query}'"}
        ]
    )
    expanded_terms = response['choices'][0]['message']['content'].strip()
    expanded_query = f"{query} {expanded_terms}"
    return expanded_query


# Function to calculate cosine similarity
def calculate_cosine_similarity(query_embedding, abstract_embedding):
    return cosine_similarity([query_embedding], [abstract_embedding])[0][0]


# Function to get the top 5 articles based on model score, combining original and expanded queries
def get_top_5_articles(query, index, top_k=5):
    # Retrieve articles and scores for the original query
    original_articles = retrieve_articles(query, top_k)
    original_scores = evaluate_articles(query, original_articles)

    # Retrieve articles and scores for the expanded query
    expanded_query = expand_query(query)
    expanded_articles = retrieve_articles(expanded_query, top_k)
    expanded_scores = evaluate_articles(query, expanded_articles)

    # Combine results from original and expanded queries
    combined_articles = []
    for i, (title, abstract) in enumerate(original_articles):
        combined_articles.append({
            "title": title,
            "abstract": abstract,
            "model_score": original_scores[i],
            "source_query": "original"
        })

    for i, (title, abstract) in enumerate(expanded_articles):
        # Avoid duplicates
        if title not in [article["title"] for article in combined_articles]:
            combined_articles.append({
                "title": title,
                "abstract": abstract,
                "model_score": expanded_scores[i],
                "source_query": "expanded"
            })

    # Sort combined articles by model score and select the top 5
    top_5_articles = sorted(combined_articles, key=lambda x: x["model_score"], reverse=True)[:5]
    return top_5_articles
