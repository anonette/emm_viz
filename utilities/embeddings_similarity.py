import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def clean_token(token):
    """Remove ## prefix and join subwords"""
    return token.replace(' ##', '')

def find_and_deduplicate_embeddings(input_ids, embeddings, all_embeddings, tokenizer, threshold=0.5):
    # Convert input IDs to tokens and clean them
    tokens = [clean_token(tokenizer.decode(id_)) for id_ in input_ids[0]]
    
    # Initialize lists for unique tokens and embeddings
    unique_tokens = []
    unique_embeddings = []
    connections = []
    
    # Process each token and its embedding
    for i, (token, embedding) in enumerate(zip(tokens, embeddings[0])):
        # Skip if token is a special token
        if token in tokenizer.all_special_tokens:
            continue
            
        # Calculate similarities with existing unique embeddings
        if unique_embeddings:
            similarities = cosine_similarity([embedding], unique_embeddings)[0]
            most_similar_idx = np.argmax(similarities)
            max_similarity = similarities[most_similar_idx]
            
            if max_similarity >= threshold:
                # Add connection if similar enough
                connections.append((i, most_similar_idx))
                continue
        
        # If we get here, add as new unique token/embedding
        unique_tokens.append(token)
        unique_embeddings.append(embedding)
    
    return unique_tokens, np.array(unique_embeddings), connections 