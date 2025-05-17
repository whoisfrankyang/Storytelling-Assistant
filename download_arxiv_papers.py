import arxiv
import requests
import os
import pandas as pd
import random
import time
from tqdm import tqdm

# Define parameters
QUERY = "machine learning OR deep learning"
CATEGORY = "cs.LG"  # Machine Learning in Computer Science
MAX_RESULTS = 100  # Fetch extra to ensure we have enough
YEAR_CUTOFF = 2022   # Only include papers from this year onwards
OUTPUT_DIR = "arxiv_papers"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ArXiv client to handle pagination
client = arxiv.Client()

# Step 1: Fetch papers from arXiv with pagination and filter by year
def fetch_arxiv_papers(query, category, max_results, year_cutoff):
    papers = []
    start = 0
    batch_size = 100  # Fetch in smaller batches

    print(f"Fetching papers from arXiv (Only from {year_cutoff} and later)...")
    while start < max_results:
        search = arxiv.Search(
            query=f"{query} AND cat:{category}",
            max_results=batch_size,
            sort_by=arxiv.SortCriterion.SubmittedDate  # Sort by most recent first
        )
        try:
            batch_papers = list(client.results(search))
            if not batch_papers:
                print(f"Empty batch received at start={start}. Stopping.")
                break  # Stop fetching if we get an empty batch

            for paper in batch_papers:
                pub_year = paper.published.year  # Extract year
                if pub_year >= year_cutoff:
                    papers.append({
                        "title": paper.title,
                        "arxiv_id": paper.entry_id.split('/')[-1],
                        "url": paper.entry_id,
                        "year": pub_year  # Store the publication year
                    })
            
            start += batch_size
            time.sleep(2)  # Avoid hitting rate limits

        except Exception as e:
            print(f"Error fetching batch {start}: {e}")
            time.sleep(5)  # Wait and retry

    return pd.DataFrame(papers)

df = fetch_arxiv_papers(QUERY, CATEGORY, MAX_RESULTS, YEAR_CUTOFF)
df_filtered = df[df["year"] >= YEAR_CUTOFF]
final_selection = df_filtered

print(f"Downloading papers from {YEAR_CUTOFF} and later...")
for _, row in tqdm(final_selection.iterrows(), total=len(final_selection)):
    arxiv_id = row["arxiv_id"]
    
    # Download PDF
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    pdf_filename = os.path.join(OUTPUT_DIR, f"{row['title'].replace(' ', '_').replace('/', '_')}.pdf")
    
    pdf_response = requests.get(pdf_url)
    if pdf_response.status_code == 200:
        with open(pdf_filename, "wb") as f:
            f.write(pdf_response.content)

print(f"Downloaded {len(final_selection)} papers from {YEAR_CUTOFF} and later. PDFs in {OUTPUT_DIR}/, HTML in {HTML_DIR}/.")
