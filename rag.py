"""
Takes the user query and retrieves the top k most relevant texts 
from the FAISS knowledge base.
"""

# Imports
import numpy as np
import pickle
import faiss
import os

# Globals
knowledge_save_path = os.path.join(
    os.path.dirname(__file__), 
    'knowledge'
)
faiss_index = faiss.read_index(
    os.path.join(
        knowledge_save_path, 
        'index.faiss'
    )
)

# Load the texts from the knowledge base
text_path = os.path.join(knowledge_save_path, "texts.pkl")
with open(text_path, "rb") as f:
    texts = pickle.load(f)

# Define consistent context prompt format
PROMPT_TEMPLATE = """
Based only on the following medical context, provide a concise 
clinical insight to help a doctor interpret a patient's symptoms. 
The insight must be medically relevant, grounded in the provided 
context, and limited to a single paragraph or short list.

{context}

---
Patient presents with the following symptoms:
{prompt}

Respond in a clear, medically appropriate tone. Do not speculate 
or provide a diagnosis.
"""

class knowledgeBase():
    """
    A simple knowledge base for storing and retrieving text 
    entries based on their embeddings and cosine similarity.

    Attributes:
        knowledge (dict): A dictionary to store entries with 
        their unique IDs, text, embeddings, and cosine 
        similarity.
    Methods:
        __len__(): Returns the number of entries in the knowledge 
            base.
        add_entry(entry_id, text, embedding): Adds an entry to 
            the knowledge base.
        get_entry_by_id(entry_id): Retrieves an entry by its ID.
        compute_cosine_similarity(vec_a, vec_b): Computes the
            cosine similarity between two normalized vectors.
        search_kb(q_embed, top_k): Searches the knowledge base 
            for the most relevant texts.
    """
    def __init__(self):
        self.knowledge = {}

    def __len__(self):
        """
        Returns the number of entries in the knowledge base.
        """
        return len(self.knowledge)

    def add_entry(
            self,
            entry_id: int,
            text: str,
            embedding: np.ndarray
    ) -> None:
        """
        Adds an entry to the knowledge base.

        Args:
            entry_id (int): Unique PubMed ID for the QA pair.
            text (str): The text content of the entry.
            embedding (np.ndarray): The embedding vector for the entry.
        """
        # Text will be the context for the question-answer pair
        self.knowledge[entry_id] = {
            'text': text,
            'embedding': embedding,
            'cosine_similarity': 0.0
        }

    def get_entry_by_id(
            self,
            entry_id: int
    ) -> dict:
        """
        Retrieves an entry from the knowledge base by its ID.

        Args:
            entry_id (int): The unique ID of the entry.
        Returns:
            dict: The entry data if found, otherwise None.
        """
        return self.knowledge.get(entry_id, None)

    def compute_cosine_similarity(
            self,
            vec_a: np.ndarray,
            vec_b: np.ndarray
    ) -> float:
        """
        Computes the cosine similarity between two normalized 
        vectors. This method assumes that the input vectors 
        have been pre-normalized through get_embedding().

        Args:
            vec_a (list or np.array): First vector.
            vec_b (list or np.array): Second vector.
        Returns:
            float: Cosine similarity between the two vectors.
        """
        return float(
            np.dot(
                vec_a.flatten(), 
                vec_b.flatten()
        ))

    def search(
            self, 
            q_embed: np.ndarray, 
            top_k: int = 3
    ) -> list[dict]:
        """
        Search the knowledge base for the most relevant texts.

        Args:
            q_embed (np.ndarray): The user query embedding.
            top_k (int): The number of top results to return.
        Returns:
            list[dict]: A list of dictionaries containing the 
                most relevant texts.
        """
        # Compute cosine similarity for every kb entry
        for entry_id, entry_data in self.knowledge.items():
            embedding = entry_data['embedding']

            # Calculate and populate cosine similarity
            cosine_similarity = self.compute_cosine_similarity(q_embed, embedding)
            self.knowledge[entry_id]['cosine_similarity'] = cosine_similarity

        # Sort kb in descending order of cosine similarity
        sorted_entries = sorted(
            self.knowledge.values(), 
            key=lambda entry: entry['cosine_similarity'], 
            reverse=True
        )

        # Return the top_k most relevant entries
        return sorted_entries[:top_k]

def load_kb(file_path: str) -> knowledgeBase:
    """
    Load a knowledge base from a pickle file.

    Args:
        file_path (str): The path to the pickle file 
            containing the knowledge base.
    Returns:
        knowledgeBase: An instance of the knowledgeBase 
            class containing the loaded data.
    """
    try:
        with open(file_path, "rb") as f:
            kb = pickle.load(f)
        return kb

    except FileNotFoundError:
        print(f"File {file_path} not found.")

def get_context_prompt(
        user_query: str, 
        results: list[dict]
) -> str:
    """
    Create a formatted context prompt for the model.

    Args:
        user_query (str): The user's query.
        results (list[dict]): The top k results from the 
            knowledge base.
    Returns:
        str: The formatted context prompt.
    """
    context = "\n".join([
        f"**{result['text']}**\n" 
        for result in results
    ])

    return PROMPT_TEMPLATE.format(
        context=context, 
        prompt=user_query
    )