# Paper Semantic Search

Find similar papers using semantic search. Supports both local models (free) and OpenAI API (better quality).

## Features

- Request for papers from OpenReview (e.g., ICLR2026 submissions)
- Semantic search with example papers or text queries
- Support embedding caching 
- Embed model support: Open-source (e.g., all-MiniLM-L6-v2) or OpenAI

## Zero-Code Quickstart

1. **Install Python 3.9 or newer.** If you are on Windows, install it from [python.org](https://www.python.org/downloads/) and enable the "Add python.exe to PATH" option.
2. **Download this project** (click the green *Code* button → *Download ZIP*) and unzip it somewhere easy to find.
3. **Open a terminal** inside the project folder and install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the one-command search experience.** Replace the file name with your own JSON/JSONL/CSV dataset (Scopus exports work out of the box):

   ```bash
   python search.py my_papers.csv --format csv --model local
   ```

   - The program will guide you with friendly prompts: simply type what you want to find (e.g. "service robots empathy"), press Enter, and the matching papers will appear.
   - Add `--save results.json` if you want the results saved automatically.
   - When your file has an obvious extension (e.g. `.csv`, `.json`), the format is detected automatically and `--format` can be skipped.

5. **(Optional) Use OpenAI embeddings** for higher quality results:

   ```bash
   setx OPENAI_API_KEY "your-key"   # Windows PowerShell
   # export OPENAI_API_KEY="your-key"  # macOS/Linux
   python search.py my_papers.csv --format csv --model openai
   ```

   You can also pass the key directly with `--api-key YOUR_KEY` when running the command.

## Working with Data

- **Crawling OpenReview papers:**

  ```python
  from crawl import crawl_papers

  crawl_papers(
      venue_id="ICLR.cc/2026/Conference/Submission",
      output_file="iclr2026_papers.json"
  )
  ```

- **Loading Scopus CSV exports:** No extra steps are required. Run `python search.py your_export.csv --format csv` or create a `PaperSearcher` manually if you prefer to integrate it into your own script. Custom column names can be provided via the `csv_mapping` parameter when they differ from the standard Scopus headers.

## Searching from Python (advanced)

```python
from search import PaperSearcher

# Local model (free)
searcher = PaperSearcher('iclr2026_papers.json', model_type='local')

# OpenAI model (better, requires API key)
# searcher = PaperSearcher('iclr2026_papers.json', model_type='openai')

searcher.compute_embeddings()

# Search with example papers that you are interested in
examples = [
    {
        "title": "Your paper title",
        "abstract": "Your paper abstract..."
    }
]

results = searcher.search(examples=examples, top_k=100)

# Or search with text query
results = searcher.search(query="interesting topics", top_k=100)

searcher.display(results, n=10)
searcher.save(results, 'results.json')
```



## How It Works

1. Paper titles and abstracts are converted to embeddings
2. Embeddings are cached automatically
3. Your query is embedded using the same model
4. Cosine similarity finds the most similar papers
5. Results are ranked by similarity score

## Cache

Embeddings are cached as `cache_<filename>_<hash>_<model>.npy`. Delete to recompute.

## Example Output

```
================================================================================
Top 100 Results (showing 10)
================================================================================

1. [0.8456] Paper a
   #12345 | foundation or frontier models, including LLMs
   https://openreview.net/forum?id=xxx

2. [0.8234] Paper b
   #12346 | applications to robotics, autonomy, planning
   https://openreview.net/forum?id=yyy
```

## Tips

- Use 1-5 example papers for best results, or a paragraph of description of your interested topic
- Local model is good enough for most cases
- OpenAI model for critical search (~$1 for 18k queries)

If it's useful, please consider giving a star~