import streamlit as st
from PIL import Image
from transformers import AutoTokenizer
import torch
from utilities.args_parser import parse_args
from utilities.embeddings_load import load_embeddings_model
from utilities.embeddings_similarity import find_and_deduplicate_embeddings
from utilities.embeddings_visualization import plot_embeddings_2d, plot_token_similarities


def main(tokenizer_name, embeddings_filename, dimensions, prompt, threshold=0.5, show_lines=False):
    # Force CPU usage
    device = torch.device('cpu')
    
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = load_embeddings_model(embeddings_filename, tokenizer.vocab_size, dimensions)
    model = model.to(device)

    # Get all embeddings for similarity comparison
    with torch.no_grad():
        all_embeddings = model.embedding.weight.cpu().numpy()

    # Create two columns for inputs
    col1, col2 = st.columns(2)

    with col1:
        # Text input for the main visualization
        prompt = st.text_input('Enter text to see word relationships:', '')
        
    with col2:
        # Single word input for similarity visualization
        word = st.text_input('Enter a word to find similar words:', '')
        
        if word:
            # Add helper message for single letters
            if len(word) == 1 and word.islower():
                st.info("""
                💡 Tip: For single letters, try:
                - Uppercase ('I' instead of 'i')
                - Add a space before the letter (' i')
                This helps BERT recognize it as a standalone letter rather than part of a word.
                """)
            
            # Show tokenization info
            tokens = tokenizer(word, return_tensors="pt")
            token_id = tokens['input_ids'][0][0]
            actual_token = tokenizer.decode([token_id])
            st.write(f"Note: '{word}' is tokenized as '{actual_token}'")

    # Show visualizations if inputs are provided
    if prompt:
        # Process and show main visualization
        tokens = tokenizer(prompt, return_tensors="pt")
        input_ids = tokens['input_ids'].to(device)
        
        with torch.no_grad():
            embeddings = model(input_ids).cpu().numpy()
        
        combined_tokens, combined_embeddings, connections = find_and_deduplicate_embeddings(
            input_ids, embeddings, all_embeddings, tokenizer, threshold)
        
        map_path = f"prompts/map_{prompt.replace(' ', '_')}.png"
        plot_embeddings_2d(tokenizer, combined_tokens, combined_embeddings, 
                       all_embeddings, show_lines, threshold, output_file=map_path)
        
        st.image(map_path, caption='Word Relationship Map', use_container_width=True)

    if word:
        # Show similar words visualization
        similarities_path = f"prompts/similarities_{word.replace(' ', '_')}.png"
        plot_token_similarities(tokenizer, word, all_embeddings, n_closest=10, 
                            output_file=similarities_path)
        
        st.image(similarities_path, caption=f'Words Similar to "{word}"', 
                use_container_width=True)

    # Add explanations in expandable sections below
    st.write("---")  # Add a separator
    
    with st.expander("How to Read These Visualizations"):
        st.write("""
        ### Word Relationship Map (Left)
        - Each point represents a word from your input text
        - Words that are closer together have more similar meanings
        - Colors show the order of words (from start to end of your text)
        - Connected words (dotted lines) are particularly similar
        
        ### Similar Words Chart (Right)
        - Shows the 10 words most similar to your chosen word
        - Longer bars indicate stronger similarity
        - Similarity scores range from -1 (opposite) to 1 (identical)
        """)

    with st.expander("Technical Details"):
        tokenizer_info = {
            'GPT-2': """
            GPT-2's tokenizer handles text by breaking it into subwords and is particularly good with:
            - Modern internet language and slang
            - Contractions (e.g., "don't", "can't")
            - Special characters and emojis
            """,
            'BERT': """
            BERT's tokenizer has some specific behaviors:
            - Treats lowercase and uppercase differently (e.g., 'i' vs 'I')
            - Single lowercase letters often get special tokenization
            - Uses '##' prefix for subwords
            """,
            'RoBERTa': """
            RoBERTa's tokenizer is an improved version of BERT's that:
            - Handles casing more consistently
            - Uses byte-level BPE tokenization
            - Better handles special characters
            """,
            'T5': """
            T5's tokenizer is designed for multiple tasks and:
            - Handles multiple languages well
            - Preserves whitespace
            - Uses sentencepiece tokenization
            """,
            'XLM-RoBERTa': """
            XLM-RoBERTa's tokenizer is optimized for multiple languages and:
            - Handles 100+ languages
            - Uses SentencePiece tokenization
            - Better with non-English characters
            """
        }
        
        st.write("""
        This tool uses word embeddings to represent words as points in a high-dimensional space.
        Similar words are positioned closer together in this space.
        """)
        
        st.write("### Current Tokenizer Details")
        st.write(tokenizer_info[selected_tokenizer])
        
        st.write("""
        ### Technical Implementation:
        - The relationship map uses PCA to reduce dimensions to 2D
        - Percentages show how much of the word relationships are captured
        - Similarity is measured using cosine similarity between word vectors
        
        ### Tips for Single Letters:
        - Try both uppercase and lowercase versions
        - Add a space before the letter (e.g., ' a' instead of 'a')
        - Different tokenizers may handle single letters differently
        """)


if __name__ == "__main__":
    # Parse the arguments
    args = parse_args()
    
    # Add tokenizer selection at the top level
    tokenizer_options = {
        'GPT-2': 'gpt2',
        'BERT': 'bert-base-uncased',
        'RoBERTa': 'roberta-base',
        'T5': 't5-base',
        'XLM-RoBERTa': 'xlm-roberta-base'
    }
    
    st.title('Word Similarity Visualization')
    st.markdown('[Visit anonette.net](https://www.anonette.net/)', unsafe_allow_html=True)
    st.write("---")  # Add a separator after the link
    
    selected_tokenizer = st.selectbox(
        'Choose a tokenizer:',
        options=list(tokenizer_options.keys()),
        help="Different tokenizers handle words differently. Try them to see the variations!"
    )
    
    # Execute with selected tokenizer
    main(tokenizer_options[selected_tokenizer], 
         args.embeddings_file, 
         args.dimensions, 
         "", # Empty initial prompt 
         0.75, 
         True)
