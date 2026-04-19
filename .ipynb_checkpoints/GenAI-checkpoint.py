import os
import re
import json
import logging
import pdfplumber
import pandas as pd
import pytesseract
from pdf2image import convert_from_path
from typing import List
from huggingface_hub import InferenceClient
import asyncio
import time

# ---------------- CONFIG ---------------- #
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

HF_TOKEN = os.getenv("HF_TOKEN")  # safer than hardcoding
client = InferenceClient(token=HF_TOKEN)

# ---------------- PDF TEXT EXTRACTION ---------------- #

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        logging.error(f"PDF extraction error: {e}")
    return text.strip()


def extract_text_with_ocr(pdf_path: str) -> str:
    logging.info("Using OCR fallback...")
    try:
        images = convert_from_path(pdf_path)
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        logging.error(f"OCR error: {e}")
        return ""

# ---------------- SMART CHUNKING ---------------- #

def chunk_text(text: str, max_chars: int = 1000) -> List[str]:
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current_chunk = [], ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# ---------------- JSON EXTRACTION ---------------- #

def extract_json(response: str) -> List[dict]:
    try:
        match = re.search(r"\[\s*{.*?}\s*]", response, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return []
    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing error: {e}")
        return []

# ---------------- LLM PROCESSING ---------------- #

def process_chunk(chunk: str, index: int, total: int, max_retries: int = 5) -> List[dict]:
    logging.info(f"Processing chunk {index+1}/{total}")

    prompt = f"""<s>[INST] You are a disaster management expert specializing in Himalayan regions.

Extract structured information about natural disasters (earthquakes, landslides, floods, avalanches, glacier bursts, etc.) in this JSON format:

[
{{
  "Disaster Type": "...",
  "Region": "...",
  "Warning Signs": "...",
  "Immediate Actions": "...",
  "Do's": "...",
  "Don'ts": "...",
  "Emergency Kit": "...",
  "Evacuation Steps": "...",
  "Communication Plan": "...",
  "Government Guidelines": "...",
  "Rescue Measures": "...",
  "Post-Disaster Actions": "..."
}}
]

Focus on:
- Causes of disasters in Himalayan regions
- Early warning signs
- What people should do before, during, and after disasters

Extract from:
{chunk} [/INST]"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="mistralai/Mistral-7B-Instruct-v0.3",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.3
            )
            output = response.choices[0].message.content
            return extract_json(output)

        except Exception as e:
            logging.warning(f"Attempt {attempt+1} failed: {e}")

            if "404" in str(e) or "not supported" in str(e).lower():
                try:
                    response = client.chat.completions.create(
                        model="meta-llama/Llama-3-8B-Instruct",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=1024,
                        temperature=0.3
                    )
                    output = response.choices[0].message.content
                    return extract_json(output)

                except Exception as fallback_e:
                    logging.warning(f"Fallback error: {fallback_e}")

            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    logging.error(f"Failed chunk {index+1}")
    return []

# ---------------- ASYNC PROCESSING ---------------- #

async def process_chunks(chunks: List[str], batch_size: int = 5) -> List[dict]:
    data = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        results = [
            process_chunk(chunk, j, len(chunks))
            for j, chunk in enumerate(batch, start=i)
        ]

        for result in results:
            if isinstance(result, list):
                data.extend(result)

        await asyncio.sleep(1)  # avoid rate limits

    return data

# ---------------- STRUCTURING ---------------- #

def structure_text_with_llm(text: str) -> pd.DataFrame:

    fields = [
        "Disaster Type", "Region", "Warning Signs", "Immediate Actions",
        "Do's", "Don'ts", "Emergency Kit", "Evacuation Steps",
        "Communication Plan", "Government Guidelines",
        "Rescue Measures", "Post-Disaster Actions"
    ]

    chunks = chunk_text(text)
    logging.info(f"Total chunks: {len(chunks)}")

    data = asyncio.run(process_chunks(chunks))

    df = pd.DataFrame(data)

    for col in fields:
        if col not in df.columns:
            df[col] = "N/A"

    return df[fields]

# ---------------- MAIN PIPELINE ---------------- #

def main(pdf_path: str):
    logging.info(f"Processing file: {pdf_path}")
    start_time = time.time()

    text = extract_text_from_pdf(pdf_path)

    if not text.strip():
        text = extract_text_with_ocr(pdf_path)

    if not text.strip():
        logging.error("No text extracted.")
        return

    logging.info(f"Sample text: {text[:200]}")

    df = structure_text_with_llm(text)

    output_csv = "himalayan_disaster_data.csv"
    output_json = "himalayan_disaster_data.json"

    df.to_csv(output_csv, index=False)
    df.to_json(output_json, orient="records", indent=2)

    logging.info(f"Saved to {output_csv} & {output_json}")
    logging.info(f"Time taken: {time.time() - start_time:.2f}s")

    print(df.head())

# ---------------- RUN ---------------- #

if __name__ == "__main__":
    pdf_path = "himalayan_disaster.pdf"  # change your file name here

    if os.path.exists(pdf_path):
        main(pdf_path)
    else:
        logging.error("PDF not found!")
        print("❌ Place your disaster PDF in this folder.")