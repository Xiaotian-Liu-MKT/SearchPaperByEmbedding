from search import PaperSearcher

# Use local model (free)
searcher = PaperSearcher('iclr2026_papers.json', model_type='local')

# Or use OpenAI (better quality)
# searcher = PaperSearcher('iclr2026_papers.json', model_type='openai')

# Or use SiliconFlow (OpenAI-compatible mainland provider)
# searcher = PaperSearcher(
#     'iclr2026_papers.json',
#     model_type='siliconflow',
#     embedding_model='BAAI/bge-large-zh-v1.5'
# )

# Or load a local Scopus CSV export
# searcher = PaperSearcher('my_scopus_export.csv', model_type='local', data_format='csv')

searcher.compute_embeddings()

examples = [
    {
        "title": "Improving Developer Emotion Classification via LLM-Based Augmentation",
        "abstract": "Detecting developer emotion in the informative data stream of technical commit messages..."
    },
]

results = searcher.search(examples=examples, top_k=100)

searcher.display(results, n=10)
searcher.save(results, 'results.json')

