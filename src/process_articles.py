import re
import hashlib
from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import pandas as pd
from datasets import Dataset
from sentence_transformers import SentenceTransformer

app = typer.Typer()
logger.add("embeddings.log", format="{time} {message}")


def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:8]


def clean_whitespace(text: str) -> str:
    # Remove newline characters
    text = text.replace('\n', ' ')
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove spaces before punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    # Remove excess spaces after punctuation
    text = re.sub(r'([.,!?;:])\s+', r'\1 ', text)
    # Strip leading and trailing spaces
    return text.strip()


def simple_sentencize(text: str) -> list:
    """
    Split text into sentences using punctuation as delimiter.
    """
    sentences = re.findall(r'[^.!?]*[.!?]', text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def chunk_sentences(sentences: list, max_tokens: int, model: SentenceTransformer) -> list:
    """
    Combine sentences into chunks so that the total token count per chunk is below `max_tokens`.
    """
    output = []
    current_chunk = []
    chunk_len = 0

    for sentence in sentences:
        tokens = model.tokenize(sentence)
        seq_len = len(tokens["input_ids"])

        if chunk_len + seq_len > max_tokens:
            # If the sentence alone is too long and current_chunk is empty,
            # split the sentence word-by-word.
            if len(current_chunk) == 0:
                parts = split_long_sentence(sentence, max_tokens=max_tokens, model=model)
                output.extend(parts)
            else:
                output.append(" ".join(current_chunk))
                current_chunk = []
                chunk_len = 0

        current_chunk.append(sentence)
        chunk_len += seq_len

    if current_chunk:
        output.append(" ".join(current_chunk))

    return output


def split_long_sentence(sentence: str, max_tokens: int, model: SentenceTransformer) -> list:
    """
    Split a long sentence into smaller parts on a word-by-word basis if its token length exceeds max_tokens.
    """
    words = sentence.split()
    parts = []
    current_part = []
    current_len = 0

    for word in words:
        tokens = model.tokenize(word)
        seq_len = len(tokens["input_ids"])

        if current_len + seq_len > max_tokens:
            parts.append(" ".join(current_part))
            current_part = []
            current_len = 0

        current_part.append(word)
        current_len += seq_len

    if current_part:
        parts.append(" ".join(current_part))

    return parts


@app.command()
def main(
    input_csv: Path = typer.Option(..., help="Path to CSV file with columns 'text' and 'article_id'"),
    output_dir: Path = typer.Option(..., help="Directory where the processed dataset will be saved, should be in embeddings"),
    model_name: str = typer.Option("MiMe-MeMo/MeMo-BERT-03", help="SentenceTransformer model name for inference"),
    max_tokens: int = typer.Option(512, help="Maximum number of tokens per chunk"),
    prefix: str = typer.Option(None, help="Optional prefix/instruction to add to each chunk before encoding"),
    prefix_description: str = typer.Option(None, help="Short description of the prefix (used in the output directory name)"),
    
):
    """
    This script reads a CSV file containing texts and their associated article IDs,
    preprocesses and chunks the texts, computes embeddings for each chunk, and saves
    the output dataset to disk.
    """
    model = SentenceTransformer(model_name)

    # Build output path based on model name and optional prefix
    mname = model_name.replace("/", "__")
    if prefix:
        if prefix_description:
            output_path = output_dir / f"emb__{mname}_{prefix_description}"
        else:
            prefix_hash = hash_prompt(prefix)
            output_path = output_dir / f"emb__{mname}_{prefix_hash}"
            logger.info(f"Hashing prefix: {prefix} == {prefix_hash}")
    else:
        output_path = output_dir / f"raw_output/emb__{mname}"

    # Read CSV into DataFrame
    df = pd.read_csv(input_csv)
    
    processed_articles = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing articles"):
        article_id = row['article_id']
        text = row['text']
        cat = row['clean_category']
        date = row['date']

        # Preprocessing: clean and split the text into sentences/chunks
        try:
            text_clean = clean_whitespace(text)
            sentences = simple_sentencize(text_clean)
            chunks = chunk_sentences(sentences, max_tokens=max_tokens, model=model)
        except Exception as e:
            logger.error(f"Preprocessing error for article_id {article_id}: {e}")
            continue

        # Inference: compute embeddings for each chunk
        try:
            embeddings = []
            for chunk in chunks:
                chunk_input = f"{prefix} {chunk}" if prefix else chunk
                emb = model.encode(chunk_input)
                embeddings.append(emb)
        except Exception as e:
            logger.error(f"Inference error for article_id {article_id}: {e}")
            continue

        processed_articles.append({
            "article_id": article_id,
            "date": date,
            "chunk": chunks,
            "embedding": embeddings,
            "clean_category": cat
        })

    # Export processed data as a Hugging Face dataset and save to disk
    dataset = Dataset.from_list(processed_articles)
    dataset.save_to_disk(output_path)
    print(f"Saved processed dataset to {output_path}")


if __name__ == "__main__":
    app()
