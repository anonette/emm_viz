import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Embedding visualization')
    parser.add_argument('--tokenizer', type=str, default='gpt2',
                      choices=['bert-base-uncased', 'gpt2', 'roberta-base', 't5-base', 'xlm-roberta-base'],
                      help='Name of the tokenizer to use')
    parser.add_argument('--embeddings_file', type=str, default='embeddings.pt',
                      help='Path to the embeddings file')
    parser.add_argument('--dimensions', type=int, default=768,
                      help='Dimensions of the embeddings')
    
    return parser.parse_args([])  # Empty list for Streamlit compatibility 