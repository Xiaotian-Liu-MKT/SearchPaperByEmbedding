# Paper Semantic Search

Find similar papers using semantic search. Supports local models (free), OpenAI, and SiliconFlow's OpenAI-compatible API.

## Features

- Request for papers from OpenReview (e.g., ICLR2026 submissions)
- Semantic search with example papers or text queries
- Support embedding caching 
- Embed model support: Open-source (e.g., all-MiniLM-L6-v2), OpenAI, or SiliconFlow models

## Zero-Code Quickstart

1. **Install Python 3.9 or newer.** If you are on Windows, install it from [python.org](https://www.python.org/downloads/) and enable the "Add python.exe to PATH" option.
2. **Download this project** (click the green *Code* button → *Download ZIP*) and unzip it somewhere easy to find.
3. **Open a terminal** inside the project folder and install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the one-command search experience.** Replace the file name with your own JSON/JSONL/CSV dataset (Scopus exports work out of the box):

   ```bash
   python search.py my_papers.csv --format csv
   ```

   - The program will guide you with friendly prompts: first pick the embedding provider (Local/OpenAI/SiliconFlow), then type what you want to find (e.g. "service robots empathy").
   - Pass `--model local`, `--model openai`, or `--model siliconflow` to skip the provider question next time.
   - Add `--save results.json` if you want the results saved automatically.
   - When your file has an obvious extension (e.g. `.csv`, `.json`), the format is detected automatically and `--format` can be skipped.

5. **(Optional) Use hosted embeddings** for higher quality results:

   ```bash
   setx OPENAI_API_KEY "your-openai-key"         # Windows PowerShell
   # export OPENAI_API_KEY="your-openai-key"     # macOS/Linux
   python search.py my_papers.csv --model openai

   setx SILICONFLOW_API_KEY "your-siliconflow-key"   # Windows PowerShell
   # export SILICONFLOW_API_KEY="your-siliconflow-key"  # macOS/Linux
   python search.py my_papers.csv --model siliconflow --embedding-model "BAAI/bge-large-zh-v1.5"
   ```

   You can also pass keys directly with `--api-key YOUR_KEY` and override models via `--embedding-model`. SiliconFlow defaults to the mainland-friendly base URL (`https://api.siliconflow.cn/v1`).

## Visual Interface (Streamlit)

Prefer buttons over terminals? Launch the Streamlit dashboard for a fully guided experience:

```bash
streamlit run app.py
```

The web app walks you through three steps:

1. **Upload your dataset** (`.json`, `.jsonl`, or `.csv`). CSV users can open the "高级：自定义 CSV 字段映射" panel to paste their own column mapping when headers differ from Scopus defaults.
2. **Pick an embedding provider** (Local/OpenAI/硅基流动) and optionally override API keys, base URLs, or exact embedding models.
3. **Describe the papers you need** and choose how many results to retrieve/display. The page shows rich cards with similarity scores, metadata, abstracts, and download links for the full JSON export.

All settings persist for the current browser session, so you can tweak queries without re-uploading the file.

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

# SiliconFlow model (OpenAI-compatible mainland provider)
# searcher = PaperSearcher(
#     'iclr2026_papers.json',
#     model_type='siliconflow',
#     embedding_model='BAAI/bge-large-zh-v1.5'
# )

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

Embeddings are cached as `cache_<filename>_<hash>_<provider>_<model>.npy`. Delete to recompute when you change datasets or models.

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
- SiliconFlow provides an OpenAI-compatible option that works well from mainland China. Set `SILICONFLOW_API_KEY` and optionally `--embedding-model` to try other hosted embeddings.

If it's useful, please consider giving a star~
