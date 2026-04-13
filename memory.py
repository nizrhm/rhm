import hnswlib
import numpy as np
import torch

class LongTermMemory:
    def __init__(self, x_dim, max_elements=10000):
        # x_dim is the flattened size of your input. For 4x4 Sudoku (16 tokens) 
        # with an embedding dimension of 128, x_dim = 16 * 128 = 2048
        self.x_dim = x_dim
        
        # Initialize the Hierarchical Navigable Small World (HNSW) graph
        # L2 distance is perfect for measuring embedding similarity
        self.index = hnswlib.Index(space='l2', dim=x_dim)
        self.index.init_index(max_elements=max_elements, ef_construction=200, M=16)
        
        # A lightweight dictionary to map the HNSW graph ID to the actual Z-tensor
        self.z_storage = {}
        self.current_id = 0

    def add_memory(self, x_embeddings, z_tensor):
        """Saves a successful thought process to Long-Term Memory."""
        # Flatten the 2D prompt embeddings into a 1D vector for math operations
        x_vec = x_embeddings.detach().flatten().cpu().numpy()
        
        # Add to the C++ HNSW graph
        self.index.add_items(x_vec, self.current_id)
        
        # Save the actual reasoning tensor in standard RAM
        self.z_storage[self.current_id] = z_tensor.detach().cpu()
        self.current_id += 1

    def recall(self, x_embeddings, k=1):
        """Instantly finds the most similar past problem and returns its reasoning."""
        if self.current_id == 0:
            return None # The AI is a newborn; it has no memories yet.
            
        x_vec = x_embeddings.detach().flatten().cpu().numpy()
        
        # Query the HNSW graph (Executes in microseconds)
        labels, distances = self.index.knn_query(x_vec, k=k)
        
        # Get the ID of the closest match
        best_id = labels[0][0]
        
        # Return the past reasoning tensor and move it back to the active device
        device = x_embeddings.device
        return self.z_storage[best_id].to(device)