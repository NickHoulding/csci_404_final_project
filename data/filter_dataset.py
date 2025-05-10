"""
Filters the dataset based on the given keywords using regex.

Usage:      1. Install python dependencies
            2. Run: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_ner_bc5cdr_md-0.4.0.tar.gz
            3. Run: python3 filter_dataset.py --help

Stops when: 1. The whole dataset is processed.
            2. The maximum matched rows are reached.

NOTE:       To 'disable' the upper limit on matched rows, set 
            maxrows to be >= the number of rows in the infile.
"""

# Imports
from spacy.lang.en import English
import concurrent.futures
import pandas as pd
import threading
import argparse
import spacy
import tqdm
import sys
import os
import re

# Adds the parent directory to the system path so the env import works
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env import get_env_var

# Globals
thread_local = threading.local()

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

def setup_args() -> argparse.Namespace:
    """
    Sets up the command line arguments for the script.

    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Filters .csv datasets using Named Entity Recognition (NER).")

    parser.add_argument("--infile", type=str, required=True,
                        help="(str) path to the existing input csv")
    parser.add_argument("--outfile", type=str, required=True,
                        help="(str) output path for the filtered csv")
    parser.add_argument("--maxrows", type=int, required=True,
                        help="(int) maximum rows in the output csv")
    parser.add_argument("--keywords", type=str, required=True,
                        help="(str) path to the .txt file with keywords, each on a new line")
    parser.add_argument("--attr", type=str, required=True,
                        help="(str) the name of the attribute to filter on")
    parser.add_argument("--ner_model", type=str, default=get_env_var('NER_MODEL'),
                        help="(str) the name of the spaCy NER model to use (defaults to NER_MODEL in env.py if not specified)")
    
    return parser.parse_args()

def validate_args(args: argparse.Namespace) -> bool:
    """
    Validates the command line arguments.

    Args:
        args (argparse.Namespace): The parsed command line arguments.
    Returns:
        bool: True if the arguments are valid, 
            False otherwise.
    """
    if args.infile == args.outfile:
        print("Arg Error: Input and output files must be different.")
        return False

    # Validate input file
    if not os.path.isfile(args.infile):
        print(f"Arg Error: Input file: \"{args.infile}\" does not exist.")
        return False
    elif not re.match(r'.*\.csv$', args.infile):
        print(f"Arg Error: Input file: \"{args.infile}\" is not a .csv file.")
        return False
    
    # Validate output file
    if not re.match(r'.*\.csv$', args.outfile):
        print(f"Arg Error: Output file: \"{args.outfile}\" is not a .csv file.")
        return False

    # Validate maxrows
    if args.maxrows <= 0:
        print("Arg Error: maxrows must be greater than 0.")
        return False

    # Validate keywords file
    if not os.path.isfile(args.keywords):
        print(f"Arg Error: Keywords file: \"{args.keywords}\" does not exist.")
        return False
    elif not re.match(r'.*\.txt$', args.keywords):
        print(f"Arg Error: Keywords file: \"{args.keywords}\" is not a .txt file.")
        return False

    # Validate attribute
    df = pd.read_csv(args.infile, nrows=0)
    if args.attr not in df.columns:
        print(f"Arg Error: Attribute: \"{args.attr}\" does not exist in the input file.")
        return False
    
    return True

def get_keywords(keywords_file: str) -> list:
    """
    Reads the keywords from a file and returns them as a list.

    Args:
        keywords_file (str): The path to the keywords file.
    Returns:
        list: A list of keywords.
    """
    with open(keywords_file, 'r') as f:
        keywords = [line.strip() for line in f.readlines()]
    
    return keywords

def get_nlp(ner_model: str) -> spacy.lang.en.English:
    """
    Loads the NER model for named entity recognition.

    Args:
        ner_model (str): The name of the NER model to load.
    Returns:
        spacy.lang.en.English: The spaCy NER model.
    """
    if not hasattr(thread_local, "nlp"):
        thread_local.nlp = spacy.load(ner_model)
    
    return thread_local.nlp

def is_respiratory_related(text: str, 
                           keywords: list, 
                           nlp=None
                           ) -> bool:
    """
    Checks if the text is related to respiratory diseases using NER.

    Args:
        text (str): The text to check.
        keywords (list): List of respiratory-related keywords.
        nlp (spacy.lang.en.English, optional): Pre-loaded spaCy model.
    Returns:
        bool: True if the text is related to respiratory diseases, 
            False otherwise.
    """
    if not isinstance(text, str):
        return False

    if nlp is None:
        nlp = get_nlp()
        
    text = text.lower()
    doc = nlp(text)
    entities = [ent.text.lower() 
                for ent in doc.ents 
                if ent.label_ in ["DISEASE", "DISEASE_OR_SYNDROME"]]
    
    for e in entities:
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw.lower()) + r'\b', e):
                return True
    
    return False

def process_chunk(chunk, 
                  processed_counter: Counter,
                  found_counter: Counter,
                  progress_bar: tqdm.tqdm,
                  max_rows: int,
                  keywords: list,
                  ner_model: str
                  ) -> pd.DataFrame:
    """
    Filters a chunk of the DataFrame.

    Args:
        chunk (pd.DataFrame): The chunk of DataFrame to filter.
        processed_counter (Counter): Counter for processed rows.
        found_counter (Counter): Counter for matched rows.
        progress_bar (tqdm.tqdm): Progress bar reference.
        max_rows (int): Maximum rows to include in the result.
        keywords (list): List of keywords to match.
    Returns:
        pd.DataFrame: A DataFrame containing the filtered rows.
    """
    results = []
    nlp = get_nlp(ner_model)

    for _, row in chunk.iterrows():
        should_process = found_counter.get() < max_rows
        
        try:
            context = row[args.attr]
            
            if should_process and not pd.isna(context) and \
                is_respiratory_related(str(context), keywords, nlp):
                results.append(row)
                found_counter.increment()
                
            processed_counter.increment()
            progress_bar.update(1)
            
        except Exception as e:
            print(f"Error processing row: {e}")
    
    return pd.DataFrame(results) if results else None

def parallel_filter_dataset(df: pd.DataFrame, 
                            num_workers: int, 
                            chunk_size: int, 
                            max_rows: int,
                            keywords: list,
                            ner_model: str
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
        i = 0

        while i < len(chunks) and found_counter.get() < max_rows:
            future = executor.submit(
                process_chunk,
                chunks[i],
                processed_counter,
                found_counter,
                progress_bar,
                max_rows,
                keywords,
                ner_model
            )
            futures.append(future)
            i += 1

        for future in concurrent.futures.as_completed(futures):
            if found_counter.get() >= max_rows:
                for f in futures:
                    if not f.done():
                        f.cancel()
        
        all_results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()

                if result is not None and not result.empty:
                    all_results.append(result)
            
            except Exception as e:
                print(f"Error collecting results: {e}")
        
        if all_results:
            with results_lock:
                filtered_df = pd.concat(all_results, ignore_index=True)
                
                if len(filtered_df) > max_rows:
                    filtered_df = filtered_df.head(max_rows)
    
    progress_bar.close()
    
    print(f"Processed {processed_counter.get()} samples")
    print(f"Found {len(filtered_df)} respiratory-related samples")
    
    return filtered_df

# Entry point
if __name__ == "__main__":
    args = setup_args()

    if not validate_args(args):
        sys.exit(1)

    keywords = get_keywords(args.keywords)
    df = pd.read_csv(args.infile)
    print(f"Loaded {len(df)} samples")
    
    num_workers = min(os.cpu_count(), 8)
    chunk_size = max(1, len(df) // (num_workers * 4))
    num_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size else 0)
    print(f"Split dataset into {num_chunks} chunks of size {chunk_size}")

    print(f"Using {num_workers} workers for parallel processing")
    filtered_df = parallel_filter_dataset(
        df,
        num_workers, 
        chunk_size, 
        args.maxrows,
        keywords,
        args.ner_model
    )
    
    print(f"Saving filtered dataset to {args.outfile}")
    out_dirname = os.path.dirname(args.outfile)
    out_path = out_dirname if out_dirname else os.getcwd()
    os.makedirs(out_path, exist_ok=True)
    filtered_df.to_csv(args.outfile, index=False)
    print(f"Saved {len(filtered_df)} samples")

    print("Filtering complete")