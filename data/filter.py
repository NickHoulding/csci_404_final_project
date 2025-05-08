"""
Filters the dataset based on the given keywords using regex.

Usage:
    1. Install dependencies
    2. Run: cd data
    3. Update the below global variables as per your needs:
        - INPUT_DATASET
        - OUTPUT_DATASET
        - MAX_ROWS
        - KEYWORDS
    4. Run: python3 filter.py

The program stops when either:
    1. The whole dataset is processed.
    2. MAX_ROWS rows have been included in OUTPUT_DATASET.

NOTE:   Your current working directory must be the same as 
        this script's directory.
"""

# Imports
import concurrent.futures
import pandas as pd
import threading
import tqdm
import os
import re

# Globals
INPUT_DATASET = "PubMedQA.csv"
OUTPUT_DATASET = "Respiratory_Small_PubMedQA.csv"
MAX_ROWS = 2000
KEYWORDS = [
    "asthma", 
    "bronchitis", 
    "pneumonia", 
    "tuberculosis", 
    "flu", 
    "influenza",
    "covid", 
    "covid-19", 
    "coronavirus", 
    "respiratory infection", 
    "lung disease", 
    "shortness of breath", 
    "dyspnea", 
    "wheezing", 
    "cough", 
    "chest tightness", 
    "pulmonary", 
    "pleurisy", 
    "ARDS", 
    "COPD", 
    "upper respiratory", 
    "lower respiratory"
]

class Counter:
    """
    A thread-safe counter class.
    
    Attributes:
        - value (int): The current count.
        - lock (threading.Lock): The thread lock.
    Methods:
        - increment(): Increments the counter by 1 and returns 
            the new value.
        - get(): Returns the current count.
    """
    def __init__(self):
        """Initializes the counter object."""
        self.value = 0
        self.lock = threading.Lock()
        
    def increment(self):
        """
        Increments the counter by 1 and returns the new value.

        Returns:
            int: The new value of the counter.
        """
        with self.lock:
            self.value += 1
            return self.value
            
    def get(self):
        """
        Returns the current count.
        
        Returns:
            int: The current count.
        """
        with self.lock:
            return self.value

def contains_respiratory_keyword(text: str) -> bool:
    """
    Checks for matching keywords using regex.
    
    Args:
        text (str): The text to check.
    Returns:
        bool: True if there are keyword matches in the text, 
            False otherwise.
    """
    if not isinstance(text, str):
        return False
        
    text = text.lower()
    for kw in KEYWORDS:
        if re.search(r'\b' + re.escape(kw.lower()) + r'\b', text):
            return True
    return False

def process_chunk(chunk, 
                  processed_counter: Counter,
                  found_counter: Counter,
                  progress_bar: tqdm.tqdm
                  ) -> pd.DataFrame:
    """
    Filters a chunk of the DataFrame.

    Args:
        chunk (pd.DataFrame): The chunk of DataFrame to filter.
        processed_counter (Counter): Counter for processed rows.
        found_counter (Counter): Counter for matched rows.
        progress_bar (tqdm.tqdm): Progress bar reference.
    Returns:
        pd.DataFrame: A DataFrame containing the filtered rows.
    """
    results = []

    for _, row in chunk.iterrows():
        try:
            context = row["context"]
            
            if pd.isna(context):
                continue
            
            if contains_respiratory_keyword(str(context)):
                results.append(row)
                found_counter.increment()
            
            processed_counter.increment()
            progress_bar.update(1)
            
            if found_counter.get() >= MAX_ROWS:
                return pd.DataFrame(results) if results else None
                
        except Exception as e:
            print(f"Error processing row: {e}")
    
    return pd.DataFrame(results) if results else None

def parallel_filter_dataset(df: pd.DataFrame, 
                            num_workers: int, 
                            chunk_size: int, 
                            max_rows: int
                            ) -> pd.DataFrame:
    """
    Filters the dataset in parallel.
    
    Args:
        df (pd.DataFrame): The input DataFrame to process.
        num_workers (int): Number of worker threads to use.
        chunk_size (int): Size of each chunk to process.
        max_rows (int): Maximum rows to include in the result.
    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    results_lock = threading.Lock()
    filtered_df = pd.DataFrame(columns=df.columns)
    
    processed_counter = Counter()
    found_counter = Counter()
    progress_bar = tqdm.tqdm(total=len(df), desc="Processing")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []

        for chunk in chunks:
            future = executor.submit(
                process_chunk,
                chunk,
                processed_counter,
                found_counter,
                progress_bar
            )
            futures.append(future)
            
            if found_counter.get() >= max_rows:
                for f in futures:
                    f.cancel()
                break
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                
                if result is not None and not result.empty:
                    with results_lock:
                        filtered_df = pd.concat(
                            [filtered_df, result], 
                            ignore_index=True
                        )
                        
                        if len(filtered_df) > max_rows:
                            filtered_df = filtered_df.head(max_rows)
                            break
            
            except Exception as e:
                print(f"Error collecting results: {e}")
    
    progress_bar.close()
    
    print(f"Processed {processed_counter.get()} samples")
    print(f"Found {len(filtered_df)} respiratory-related samples")
    
    return filtered_df

# Entry point
if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    input_path = os.path.join(current_dir, INPUT_DATASET)
    output_path = os.path.join(current_dir, OUTPUT_DATASET)

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} samples")
    
    num_workers = min(os.cpu_count(), 8)
    chunk_size = max(1, len(df) // (num_workers * 4))
    num_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size else 0)
    print(f"Split dataset into {num_chunks} chunks of size {chunk_size}")

    print(f"Using {num_workers} workers for parallel processing")
    filtered_df = parallel_filter_dataset(df, num_workers, chunk_size, MAX_ROWS)
    
    print(f"Saving filtered dataset to {output_path}")
    filtered_df.to_csv(output_path, index=False)
    print(f"Saved {len(filtered_df)} samples")

    print("Filtering complete")