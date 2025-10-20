import argparse
import csv
import json
import numpy as np
import os
import sys
import hashlib
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
class PaperSearcher:
    def __init__(
        self,
        papers_file,
        model_type="openai",
        api_key=None,
        base_url=None,
        data_format=None,
        csv_mapping=None,
        embedding_model=None,
    ):
        self.papers_file = papers_file
        self.data_format = (data_format or Path(papers_file).suffix.lower().lstrip('.') or 'json')
        self.csv_mapping = csv_mapping or {}
        self.papers = self._load_papers(papers_file)
        self.model_type = model_type
        self.embeddings = None
        if model_type in {"openai", "siliconflow"}:
            from openai import OpenAI
            if model_type == "openai":
                resolved_api_key = api_key or os.getenv('OPENAI_API_KEY')
                resolved_base_url = base_url
                default_model = "text-embedding-3-large"
            else:
                resolved_api_key = api_key or os.getenv('SILICONFLOW_API_KEY')
                resolved_base_url = base_url or "https://api.siliconflow.cn/v1"
                default_model = "BAAI/bge-large-zh-v1.5"
            if not resolved_api_key:
                env_hint = 'OPENAI_API_KEY' if model_type == "openai" else 'SILICONFLOW_API_KEY'
                raise ValueError(
                    f"API key is missing. Pass it with api_key=... or set the {env_hint} environment variable."
                )
            client_kwargs = {'api_key': resolved_api_key}
            if resolved_base_url:
                client_kwargs['base_url'] = resolved_base_url
            self.client = OpenAI(**client_kwargs)
            self.model_name = embedding_model or default_model
        elif model_type == "local":
            from sentence_transformers import SentenceTransformer
            local_model_name = embedding_model or 'all-MiniLM-L6-v2'
            self.model = SentenceTransformer(local_model_name)
            self.model_name = local_model_name
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        self.cache_file = self._get_cache_file(papers_file, model_type, self.model_name)
        self._load_cache()
    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _load_papers(self, papers_file):
        if self.data_format in ("json", "jsonl"):
            with open(papers_file, 'r', encoding='utf-8') as f:
                if self.data_format == "jsonl":
                    return [json.loads(line) for line in f if line.strip()]
                return json.load(f)
        if self.data_format == "csv":
            return self._load_csv_papers(papers_file)
        raise ValueError(f"Unsupported data format: {self.data_format}")
    def _normalize_key(self, name):
        return ''.join(ch for ch in name.lower() if not ch.isspace())
    def _load_csv_papers(self, papers_file):
        default_mapping = {
            'title': [
                'Title', 'Document Title', '文献标题'
            ],
            'abstract': ['Abstract', '摘要'],
            'keywords': [
                'Author Keywords', 'Authors Keywords', '作者关键字',
                'Index Keywords', '索引关键字'
            ],
            'authors': ['Authors', 'Author full names', '作者'],
            'year': ['Year', '年份'],
            'source': ['Source Title', '来源出版物名称'],
            'paper_id': ['EID', '论文编号'],
            'link': ['Link', '链接'],
            'doi': ['DOI']
        }
        mapping = {}
        for key, defaults in default_mapping.items():
            overrides = self.csv_mapping.get(key)
            if overrides is None:
                mapping[key] = defaults
            elif isinstance(overrides, str):
                mapping[key] = [overrides]
            else:
                mapping[key] = list(overrides)
            # ensure defaults are also present in case-insensitive manner
            for value in defaults:
                if value not in mapping[key]:
                    mapping[key].append(value)
        papers = []
        with open(papers_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                normalized_row = {
                    self._normalize_key(key): (value.strip() if isinstance(value, str) else value)
                    for key, value in row.items()
                }
                paper = {}
                title = self._extract_csv_value(normalized_row, mapping['title'])
                if not title:
                    # Skip rows without a title since the downstream pipeline expects one
                    continue
                paper['title'] = title
                abstract = self._extract_csv_value(normalized_row, mapping['abstract'])
                if abstract:
                    paper['abstract'] = abstract
                authors = self._extract_csv_list(normalized_row, mapping['authors'])
                if authors:
                    paper['authors'] = authors
                year = self._extract_csv_value(normalized_row, mapping['year'])
                if year:
                    paper['year'] = year
                source = self._extract_csv_value(normalized_row, mapping['source'])
                if source:
                    paper['source'] = source
                paper_id = self._extract_csv_value(normalized_row, mapping['paper_id'])
                if paper_id:
                    paper['number'] = paper_id
                link = self._extract_csv_value(normalized_row, mapping['link'])
                if link:
                    paper['forum_url'] = link
                doi = self._extract_csv_value(normalized_row, mapping['doi'])
                if doi:
                    paper['doi'] = doi
                keywords = self._extract_csv_list(normalized_row, mapping['keywords'])
                if keywords:
                    paper['keywords'] = keywords
                paper['raw_row'] = row
                papers.append(paper)
        if not papers:
            raise ValueError("No papers were loaded from the CSV file. Check the mapping or encoding.")
        return papers
    def _extract_csv_value(self, normalized_row, candidates):
        for candidate in candidates:
            key = self._normalize_key(candidate)
            if key in normalized_row:
                value = normalized_row[key]
                if isinstance(value, str):
                    value = value.strip()
                if value:
                    return value
        return ""
    def _extract_csv_list(self, normalized_row, candidates):
        raw_values = []
        for candidate in candidates:
            key = self._normalize_key(candidate)
            if key in normalized_row and normalized_row[key]:
                raw_values.append(normalized_row[key])
        items = []
        for value in raw_values:
            if not value:
                continue
            if isinstance(value, str):
                parts = [value]
                for sep in [';', '|', ',', '\u3001', '\uff1b']:
                    if sep in value:
                        parts = value.split(sep)
                        break
                items.extend(part.strip() for part in parts if part.strip())
            elif isinstance(value, list):
                items.extend(str(item).strip() for item in value if str(item).strip())
            else:
                items.append(str(value).strip())
        # Preserve order while removing duplicates
        seen = set()
        unique_items = []
        for item in items:
            if item not in seen:
                seen.add(item)
                unique_items.append(item)
        return unique_items
    def _get_cache_file(self, papers_file, model_type, model_name):
        base_name = Path(papers_file).stem
        file_hash = hashlib.md5(papers_file.encode()).hexdigest()[:8]
        safe_model = ''.join(ch if ch.isalnum() or ch in {'-', '_'} else '_' for ch in model_name)
        cache_name = f"cache_{base_name}_{file_hash}_{model_type}_{safe_model}.npy"
        return str(Path(papers_file).parent / cache_name)
    
    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                self.embeddings = np.load(self.cache_file)
                if len(self.embeddings) == len(self.papers):
                    print(f"Loaded cache: {self.embeddings.shape}")
                    return True
                self.embeddings = None
            except:
                self.embeddings = None
        return False
    
    def _save_cache(self):
        np.save(self.cache_file, self.embeddings)
        print(f"Saved cache: {self.cache_file}")
    
    def _create_text(self, paper):
        parts = []
        if paper.get('title'):
            parts.append(f"Title: {paper['title']}")
        if paper.get('authors'):
            authors = ', '.join(paper['authors']) if isinstance(paper['authors'], list) else paper['authors']
            parts.append(f"Authors: {authors}")
        if paper.get('year'):
            parts.append(f"Year: {paper['year']}")
        if paper.get('source'):
            parts.append(f"Source: {paper['source']}")
        if paper.get('abstract'):
            parts.append(f"Abstract: {paper['abstract']}")
        if paper.get('keywords'):
            kw = ', '.join(paper['keywords']) if isinstance(paper['keywords'], list) else paper['keywords']
            parts.append(f"Keywords: {kw}")
        return ' '.join(parts)
    
    def _embed_openai(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(input=batch, model=self.model_name)
            embeddings.extend([item.embedding for item in response.data])
        
        return np.array(embeddings)
    
    def _embed_local(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, show_progress_bar=len(texts) > 100)
    
    def compute_embeddings(self, force=False):
        if self.embeddings is not None and not force:
            print("Using cached embeddings")
            return self.embeddings
        
        print(f"Computing embeddings ({self.model_name})...")
        texts = [self._create_text(p) for p in self.papers]
        
        if self.model_type in {"openai", "siliconflow"}:
            self.embeddings = self._embed_openai(texts)
        else:
            self.embeddings = self._embed_local(texts)
        
        print(f"Computed: {self.embeddings.shape}")
        self._save_cache()
        return self.embeddings
    
    def search(self, examples=None, query=None, top_k=100):
        if self.embeddings is None:
            self.compute_embeddings()
        
        if examples:
            texts = []
            for ex in examples:
                text = f"Title: {ex['title']}"
                if ex.get('abstract'):
                    text += f" Abstract: {ex['abstract']}"
                texts.append(text)
            
            if self.model_type in {"openai", "siliconflow"}:
                embs = self._embed_openai(texts)
            else:
                embs = self._embed_local(texts)

            query_emb = np.mean(embs, axis=0).reshape(1, -1)

        elif query:
            if self.model_type in {"openai", "siliconflow"}:
                query_emb = self._embed_openai(query).reshape(1, -1)
            else:
                query_emb = self._embed_local(query).reshape(1, -1)
        else:
            raise ValueError("Provide either examples or query")
        
        similarities = cosine_similarity(query_emb, self.embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [{
            'paper': self.papers[idx],
            'similarity': float(similarities[idx])
        } for idx in top_indices]
    
    def display(self, results, n=10):
        print(f"\n{'='*80}")
        print(f"Top {len(results)} Results (showing {min(n, len(results))})")
        print(f"{'='*80}\n")
        for i, result in enumerate(results[:n], 1):
            paper = result['paper']
            sim = result['similarity']
            print(f"{i}. [{sim:.4f}] {paper['title']}")
            number = paper.get('number') or paper.get('paper_id') or paper.get('eid') or 'N/A'
            area = paper.get('primary_area') or paper.get('source') or paper.get('journal') or 'N/A'
            link = paper.get('forum_url') or paper.get('link') or paper.get('url') or paper.get('doi') or 'N/A'
            print(f"   #{number} | {area}")
            print(f"   {link}\n")
    
    def save(self, results, output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'model': self.model_name,
                'total': len(results),
                'results': results
            }, f, ensure_ascii=False, indent=2)
        print(f"Saved to {output_file}")
def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Search similar research papers without writing code.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "data",
        help="Path to your paper dataset (.json, .jsonl, or .csv such as a Scopus export)."
    )
    parser.add_argument(
        "--format",
        choices=["json", "jsonl", "csv"],
        help="Format of the input data. Detected automatically from the file extension when omitted."
    )
    parser.add_argument(
        "--model",
        choices=["local", "openai", "siliconflow"],
        help="Embedding provider to use. The local option runs fully offline.",
    )
    parser.add_argument(
        "--api-key",
        help="API key for OpenAI or SiliconFlow (defaults to OPENAI_API_KEY / SILICONFLOW_API_KEY env vars).",
    )
    parser.add_argument(
        "--base-url",
        help="Optional custom OpenAI-compatible API base URL (defaults to SiliconFlow when using that provider)."
    )
    parser.add_argument(
        "--embedding-model",
        help="Custom embedding model identifier (for example text-embedding-3-large or BAAI/bge-large-zh-v1.5).",
    )
    parser.add_argument(
        "--query",
        help="Text description of the papers you are looking for. If omitted, you will be prompted interactively."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many similar papers to retrieve."
    )
    parser.add_argument(
        "--show",
        type=int,
        help="How many results to show in the terminal. Defaults to the same number as --top-k."
    )
    parser.add_argument(
        "--save",
        help="Optional path to save the full results as JSON."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute embeddings even if a cache file already exists."
    )
    return parser.parse_args(argv)


def _prompt_query():
    print("\nWhat kind of papers would you like to find?")
    print("Type a few keywords or a short description, then press Enter.")
    while True:
        query = input("> ").strip()
        if query:
            return query
        print("Please enter at least a few words describing the papers you need.")


def _prompt_model():
    print("\nChoose an embedding provider:")
    print("1) Local (free, runs on your computer)")
    print("2) OpenAI (requires an OpenAI API key)")
    print("3) SiliconFlow (requires a SiliconFlow API key)")

    mapping = {"1": "local", "2": "openai", "3": "siliconflow"}
    while True:
        choice = input("Select 1, 2, or 3 [1]: ").strip()
        if not choice:
            return "local"
        if choice in mapping:
            return mapping[choice]
        print("Please type 1, 2, or 3.")


def main(argv=None):
    args = _parse_args(argv)

    if args.model is None:
        args.model = _prompt_model()

    if args.model == "siliconflow" and not args.base_url:
        args.base_url = "https://api.siliconflow.cn/v1"

    try:
        searcher = PaperSearcher(
            args.data,
            model_type=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            data_format=args.format,
            embedding_model=args.embedding_model,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    try:
        searcher.compute_embeddings(force=args.force)
    except Exception as exc:
        print(f"Failed to compute embeddings: {exc}")
        return 1
    query = args.query.strip() if args.query else _prompt_query()
    try:
        results = searcher.search(query=query, top_k=args.top_k)
    except Exception as exc:
        print(f"Search failed: {exc}")
        return 1
    display_n = args.show if args.show is not None else min(args.top_k, 10)
    searcher.display(results, n=display_n)
    if args.save:
        try:
            searcher.save(results, args.save)
        except Exception as exc:
            print(f"Could not save results: {exc}")
            return 1
    print("Done! You can rerun this command with a different --query to explore new topics.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
