import torch
import torch.nn as nn

class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(EmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
    
    def forward(self, x):
        return self.embedding(x)

def load_embeddings_model(embeddings_file, vocab_size, embedding_dim):
    model = EmbeddingModel(vocab_size, embedding_dim)
    try:
        # Use weights_only=True to avoid pickle security warning
        model.load_state_dict(torch.load(embeddings_file, 
                                       map_location=torch.device('cpu'),
                                       weights_only=True))
    except FileNotFoundError:
        print(f"No pre-trained embeddings found at {embeddings_file}. Using random initialization.")
    except Exception as e:
        print(f"Error loading embeddings: {e}. Using random initialization.")
    
    model.eval()  # Set to evaluation mode
    return model 