import numpy as np
import pinecone
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Initialize Pinecone and create/connect to an index
def initialize_pinecone(api_key, index_name, environment='us-east-1'):
    # Initialize Pinecone
    pinecone.init(api_key=api_key, environment=environment)

    # Check if the index exists and create it if it does not
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            dimension=384,  # Ensure this matches your embedding dimension
            metric='cosine'
        )

    # Connect to the existing or newly created index
    index = pinecone.Index(index_name)
    return index


def load_models():
    full_model = SentenceTransformer('bert-base-uncased')
    mini_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return full_model, mini_model


# Create category embeddings
def create_category_embeddings(model, category_dict):
    category_embedding_dict = {}
    for code, name in category_dict.items():
        embedding = model.encode(name)
        category_embedding_dict[code] = {
            'name': name,
            'embedding': embedding
        }
    return category_embedding_dict


# Upsert document embeddings into Pinecone
def upsert_document_embeddings(index, document_embedding_dict, batch_size=100):
    embeddings_batch = []
    for i, (doc_id, data) in enumerate(document_embedding_dict.items()):
        embedding = data['embedding']
        metadata = {
            'categories': data['categories'],
            'update_date': str(data['update_date']),
            'title': data.get('title', ''),
            'abstract': data.get('abstract', '')
        }
        embeddings_batch.append((doc_id, embedding.tolist(), metadata))

        if (i + 1) % batch_size == 0 or i == len(document_embedding_dict) - 1:
            index.upsert(embeddings_batch)
            embeddings_batch = []


# Query handling and category-aware retrieval
def find_top_categories(query, full_model, category_embedding_dict):
    query_embedding_full = full_model.encode(query)
    category_embeddings = np.array([data['embedding'] for data in category_embedding_dict.values()])
    category_codes = list(category_embedding_dict.keys())
    similarities = cosine_similarity([query_embedding_full], category_embeddings)[0]
    top_3_indices = similarities.argsort()[-3:][::-1]
    top_3_categories = [category_codes[i] for i in top_3_indices]
    return top_3_categories


def retrieve_documents(index, query_embedding, top_3_categories):
    filtered_documents = []
    for category in top_3_categories:
        # Query the index with a filter for the specific category
        query_result = index.query(
            vector=query_embedding.tolist(),
            top_k=5,
            include_values=True,
            include_metadata=True,
            filter={"categories": {"$in": [category]}}  # Filter to include only documents in the current category
        )
        if 'matches' in query_result and len(query_result['matches']) > 0:
            filtered_documents.extend(query_result['matches'])
    return filtered_documents


def get_top_unique_documents(filtered_documents):
    if len(filtered_documents) > 0:
        sorted_documents = sorted(filtered_documents, key=lambda x: x['score'], reverse=True)
        unique_docs = []
        seen_doc_ids = set()

        for match in sorted_documents:
            doc_id = match['id']
            if doc_id not in seen_doc_ids:
                unique_docs.append(match)
                seen_doc_ids.add(doc_id)
            if len(unique_docs) == 5:
                break

        return unique_docs
    else:
        return []


def display_documents(unique_docs):
    if unique_docs:
        print("Top 5 most relevant unique documents:")
        for match in unique_docs:
            doc_id = match['id']
            metadata = match.get('metadata', {})
            title = metadata.get('title', 'No title available')
            abstract = metadata.get('abstract', 'No abstract available')
            score = match.get('score', 'No score available')
            print(f"Document ID: {doc_id}\nTitle: {title}\nAbstract: {abstract}\nScore: {score}\n")
    else:
        print("No relevant documents found.")


def main():

    pinecone.init(api_key="65adfe61-8c99-4c68-951e-e2d42e7884df", environment="us-east-1")
    index = pinecone.Index("document-embeddings")

    # load BERT models
    full_model, mini_model = load_models()

    # Create category embeddings
    category_dict = {
        'cs.AI': 'Artificial Intelligence',
        'cs.LG': 'Machine Learning',
        'cs.CL': 'Computation and Language',
        'cs.CV': 'Computer Vision and Pattern Recognition',
        'stat.ML': 'Machine Learning',
        'cs.NE': 'Neural and Evolutionary Computing',
        'eess.AS': 'Audio and Speech Processing',
        'stat.TH': 'Statistics Theory'
    }
    category_embedding_dict = create_category_embeddings(full_model, category_dict)

    # Query handling and retrieval
    query = "deep learning"
    query_embedding = mini_model.encode(query)
    top_3_categories = find_top_categories(query, full_model, category_embedding_dict)

    print("Top 3 categories for the query:")
    for category_code in top_3_categories:
        print(f"{category_code} : {category_dict[category_code]}")

    # Step 4: Retrieve relevant documents
    filtered_documents = retrieve_documents(index, query_embedding, top_3_categories)
    unique_docs = get_top_unique_documents(filtered_documents)

    # Step 5: Display results
    display_documents(unique_docs)


if __name__ == "__main__":
    main()
