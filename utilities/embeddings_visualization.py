import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

def clean_token(token):
    """Remove ## prefix and join subwords"""
    return token.replace(' ##', '')

def plot_embeddings_2d(tokenizer, tokens, embeddings, all_embeddings, show_lines=True, 
                      threshold=0.5, output_file='embeddings_plot.png'):
    # Clean tokens
    tokens = [clean_token(token) for token in tokens]
    
    # Set the style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Check if we have enough samples for dimensionality reduction
    n_samples = len(embeddings)
    if n_samples < 2:
        raise ValueError("Need at least 2 tokens to create visualization")
    
    # Use PCA instead of t-SNE (much faster)
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Create the plot with a specific background color
    plt.figure(figsize=(12, 8), facecolor='#f0f2f6')
    ax = plt.gca()
    ax.set_facecolor('#f0f2f6')
    
    # Plot points with a nicer style
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=np.arange(len(tokens)),
                         cmap='viridis',
                         s=100,
                         alpha=0.6,
                         edgecolors='white',
                         linewidth=2)
    
    # Add labels with a nicer style
    for i, (token, point) in enumerate(zip(tokens, embeddings_2d)):
        plt.annotate(token, 
                    (point[0], point[1]),
                    xytext=(7, 7),
                    textcoords='offset points',
                    fontsize=14,
                    fontweight='bold',
                    bbox=dict(
                        facecolor='white',
                        edgecolor='#666666',
                        alpha=0.9,
                        pad=1.5,
                        boxstyle='round,pad=0.5'
                    ))
    
    # Optimize similarity calculations if show_lines is True
    if show_lines and n_samples > 1:
        # Vectorized similarity calculation
        norms = np.linalg.norm(embeddings, axis=1)
        similarities = np.dot(embeddings, embeddings.T) / np.outer(norms, norms)
        # Get pairs of similar tokens
        similar_pairs = np.where(np.triu(similarities > threshold, k=1))
        for i, j in zip(*similar_pairs):
            plt.plot([embeddings_2d[i, 0], embeddings_2d[j, 0]],
                    [embeddings_2d[i, 1], embeddings_2d[j, 1]],
                    'gray', alpha=0.2, linestyle='--')
    
    # Set title and labels with better styling
    var_explained = pca.explained_variance_ratio_ * 100
    plt.title(f'Word Similarity Map\n{n_samples} tokens\n' +
             f'(Captures {var_explained[0]:.1f}% and {var_explained[1]:.1f}% of word relationships)', 
              fontsize=16, 
              fontweight='bold',
              pad=20)
    plt.xlabel('Main Direction of Variation', fontsize=12, labelpad=10)
    plt.ylabel('Secondary Direction of Variation', fontsize=12, labelpad=10)
    
    # Add explanation text
    explanation = (
        "Words that are closer together are more similar in meaning.\n"
        "The horizontal axis shows the strongest pattern of relationships between words,\n"
        "while the vertical axis shows secondary patterns."
    )
    plt.figtext(0.02, 0.02, explanation, fontsize=10, alpha=0.7, wrap=True)
    
    # Add grid with better styling
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    
    # Add a subtle border
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#666666')
    ax.spines['left'].set_color('#666666')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot with high quality
    plt.savefig(output_file, 
                bbox_inches='tight', 
                dpi=300, 
                facecolor='#f0f2f6',
                edgecolor='none')
    plt.close() 

def plot_token_similarities(tokenizer, token, all_embeddings, n_closest=10, output_file='token_similarities.png'):
    # Get the token ID and show what the tokenizer actually sees
    tokens = tokenizer(token, return_tensors="pt")
    token_id = tokens['input_ids'][0][0]
    actual_token = tokenizer.decode([token_id])
    
    print(f"Input: '{token}'")
    print(f"Tokenized as: '{actual_token}'")
    
    # If the tokenized result is different from input, try to find the exact token
    if clean_token(actual_token) != token:
        # Try to find token that exactly matches input
        for i in range(len(all_embeddings)):
            if clean_token(tokenizer.decode([i])) == token:
                token_id = i
                actual_token = tokenizer.decode([i])
                break
    
    token_embedding = all_embeddings[token_id]
    
    # Calculate similarities with all tokens
    norms = np.linalg.norm(all_embeddings, axis=1)
    similarities = np.dot(all_embeddings, token_embedding) / (norms * np.linalg.norm(token_embedding))
    
    # Get top N most similar tokens (excluding the token itself)
    top_n_indices = np.argsort(similarities)[-n_closest-1:-1][::-1]
    top_n_similarities = similarities[top_n_indices]
    # Clean the tokens when decoding
    top_n_tokens = [clean_token(tokenizer.decode([i])) for i in top_n_indices]
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(15, 7), facecolor='#f0f2f6')
    
    # Bar chart on the left
    ax1 = plt.subplot(121)
    ax1.set_facecolor('#f0f2f6')
    
    # Create horizontal bar chart
    bars = ax1.barh(range(len(top_n_tokens)), top_n_similarities, 
                    color=plt.cm.viridis(np.linspace(0, 0.8, len(top_n_tokens))),
                    alpha=0.6)
    
    # Add value labels on the bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.2f}',
                ha='left', va='center', fontsize=10,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))
    
    ax1.set_title('Similarity Scores', fontsize=14, pad=20)
    ax1.set_xlabel('Cosine Similarity', fontsize=12, labelpad=10)
    ax1.set_yticks(range(len(top_n_tokens)))
    ax1.set_yticklabels(top_n_tokens, fontsize=12)
    
    # Radial visualization on the right
    ax2 = plt.subplot(122, projection='polar')
    ax2.set_facecolor('#f0f2f6')
    
    # Calculate angles and distances for radial plot
    angles = np.linspace(0, 2*np.pi, len(top_n_tokens), endpoint=False)
    # Convert similarities to distances (closer = more similar)
    distances = 1 - top_n_similarities
    
    # Plot points in radial layout
    ax2.scatter(angles, distances, c=plt.cm.viridis(np.linspace(0, 0.8, len(top_n_tokens))),
                s=200, alpha=0.6, edgecolors='white', linewidth=2)
    
    # Add labels
    for angle, distance, token in zip(angles, distances, top_n_tokens):
        ha = 'left' if 0 <= angle <= np.pi else 'right'
        ax2.text(angle, distance, f' {token} ', 
                ha=ha, va='center', fontsize=12,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=2))
    
    # Customize radial plot
    ax2.set_title('Proximity Map', fontsize=14, pad=20)
    ax2.set_rticks([0, 0.25, 0.5, 0.75])
    ax2.set_rlabel_position(0)
    ax2.set_rlim(0, 1)
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # Add main title
    plt.suptitle(f'Words Most Similar to "{clean_token(actual_token)}" (input: "{token}")', 
                 fontsize=16, fontweight='bold', y=1.05)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file, 
                bbox_inches='tight', 
                dpi=300, 
                facecolor='#f0f2f6',
                edgecolor='none')
    plt.close() 