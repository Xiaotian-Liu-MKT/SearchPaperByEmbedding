"""Utilities for computing and searching paper embeddings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config_utils import get_setting


class PaperSearcher:
    """Embed and search papers from CSV/JSON/JSONL collections."""

    def __init__(
        self,
        papers_file: str,
        model_type: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        data_format: Optional[str] = None,
        csv_mapping: Optional[Dict[str, Sequence[str]]] = None,
        embedding_model: Optional[str] = None,
    ) -> None:
        self.papers_file = str(papers_file)
        self.data_format = (data_format or Path(self.papers_file).suffix.lstrip(".") or "json").lower()
        self.csv_mapping: Dict[str, Sequence[str]] = csv_mapping or {}
        self.papers = self._load_papers(self.papers_file)

        self.model_type = model_type
        self.embeddings: Optional[np.ndarray] = None

        if model_type in {"openai", "siliconflow"}:
            from openai import OpenAI

            if model_type == "openai":
                resolved_api_key = api_key or get_setting("OPENAI_API_KEY")
                resolved_base_url = base_url
                default_model = "text-embedding-3-large"
            else:
                resolved_api_key = api_key or get_setting("SILICONFLOW_API_KEY")
                resolved_base_url = base_url or "https://api.siliconflow.cn/v1"
                default_model = "BAAI/bge-large-zh-v1.5"

            if not resolved_api_key:
                hint = "OPENAI_API_KEY" if model_type == "openai" else "SILICONFLOW_API_KEY"
                raise ValueError(
                    "API key is missing. Provide it explicitly or set it in .env/config.json under "
                    f"{hint}."
                )

            client_kwargs = {"api_key": resolved_api_key}
            if resolved_base_url:
                client_kwargs["base_url"] = resolved_base_url
            self.client = OpenAI(**client_kwargs)
            self.model_name = embedding_model or default_model
        elif model_type == "local":
            from sentence_transformers import SentenceTransformer

            local_model_name = embedding_model or "all-MiniLM-L6-v2"
            self.model = SentenceTransformer(local_model_name)
            self.model_name = local_model_name
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        self.cache_file = self._get_cache_file(self.papers_file, model_type, self.model_name)
        self._load_cache()

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------
    def _load_papers(self, papers_file: str) -> List[Dict[str, object]]:
        path = Path(papers_file)
        if self.data_format == "jsonl":
            with path.open("r", encoding="utf-8") as handle:
                return [json.loads(line) for line in handle if line.strip()]
        if self.data_format == "json":
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        if self.data_format == "csv":
            return self._load_csv_papers(path)
        raise ValueError(f"Unsupported data format: {self.data_format}")

    @staticmethod
    def _normalize_key(name: str) -> str:
        return "".join(ch for ch in name.lower() if not ch.isspace())

    def _load_csv_papers(self, path: Path) -> List[Dict[str, object]]:
        default_mapping: Dict[str, Sequence[str]] = {
            "title": ["Title", "Document Title", "文献标题"],
            "abstract": ["Abstract", "摘要"],
            "keywords": [
                "Author Keywords",
                "Authors Keywords",
                "作者关键字",
                "Index Keywords",
                "索引关键字",
            ],
            "authors": ["Authors", "Author full names", "作者"],
            "year": ["Year", "年份"],
            "source": ["Source Title", "来源出版物名称"],
            "paper_id": ["EID", "论文编号"],
            "link": ["Link", "链接"],
            "doi": ["DOI"],
        }

        mapping: Dict[str, List[str]] = {}
        for key, defaults in default_mapping.items():
            overrides = self.csv_mapping.get(key)
            if overrides is None:
                mapping[key] = list(defaults)
            elif isinstance(overrides, str):
                mapping[key] = [overrides]
            else:
                mapping[key] = list(overrides)
            for value in defaults:
                if value not in mapping[key]:
                    mapping[key].append(value)

        papers: List[Dict[str, object]] = []
        with path.open("r", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                normalized_row = {
                    self._normalize_key(key): (value.strip() if isinstance(value, str) else value)
                    for key, value in row.items()
                }
                title = self._extract_csv_value(normalized_row, mapping["title"])
                if not title:
                    continue

                paper: Dict[str, object] = {"title": title}
                abstract = self._extract_csv_value(normalized_row, mapping["abstract"])
                if abstract:
                    paper["abstract"] = abstract
                authors = self._extract_csv_list(normalized_row, mapping["authors"])
                if authors:
                    paper["authors"] = authors
                year = self._extract_csv_value(normalized_row, mapping["year"])
                if year:
                    paper["year"] = year
                source = self._extract_csv_value(normalized_row, mapping["source"])
                if source:
                    paper["source"] = source
                paper_id = self._extract_csv_value(normalized_row, mapping["paper_id"])
                if paper_id:
                    paper["number"] = paper_id
                link = self._extract_csv_value(normalized_row, mapping["link"])
                if link:
                    paper["forum_url"] = link
                doi = self._extract_csv_value(normalized_row, mapping["doi"])
                if doi:
                    paper["doi"] = doi
                keywords = self._extract_csv_list(normalized_row, mapping["keywords"])
                if keywords:
                    paper["keywords"] = keywords
                paper["raw_row"] = row
                papers.append(paper)

        if not papers:
            raise ValueError("No papers were loaded from the CSV file. Check the mapping or encoding.")
        return papers

    def _extract_csv_value(self, normalized_row: Dict[str, str], candidates: Sequence[str]) -> str:
        for candidate in candidates:
            key = self._normalize_key(candidate)
            if key in normalized_row:
                value = normalized_row[key]
                if isinstance(value, str):
                    value = value.strip()
                if value:
                    return str(value)
        return ""

    def _extract_csv_list(self, normalized_row: Dict[str, str], candidates: Sequence[str]) -> List[str]:
        raw_values: List[object] = []
        for candidate in candidates:
            key = self._normalize_key(candidate)
            value = normalized_row.get(key)
            if value:
                raw_values.append(value)

        items: List[str] = []
        for value in raw_values:
            if isinstance(value, str):
                parts = [value]
                for sep in [";", "|", ",", "\u3001", "\uff1b"]:
                    if sep in value:
                        parts = value.split(sep)
                        break
                items.extend(part.strip() for part in parts if part.strip())
            elif isinstance(value, list):
                items.extend(str(item).strip() for item in value if str(item).strip())
            else:
                items.append(str(value).strip())

        seen = set()
        unique_items: List[str] = []
        for item in items:
            if item not in seen:
                seen.add(item)
                unique_items.append(item)
        return unique_items

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _get_cache_file(papers_file: str, model_type: str, model_name: str) -> str:
        base_name = Path(papers_file).stem
        file_hash = hashlib.md5(papers_file.encode()).hexdigest()[:8]
        safe_model = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in model_name)
        cache_name = f"cache_{base_name}_{file_hash}_{model_type}_{safe_model}.npy"
        return str(Path(papers_file).parent / cache_name)

    def _load_cache(self) -> bool:
        if os.path.exists(self.cache_file):
            try:
                embeddings = np.load(self.cache_file)
            except Exception:  # pragma: no cover - cache corruption fallback
                return False
            if len(embeddings) == len(self.papers):
                self.embeddings = embeddings
                return True
        return False

    def _save_cache(self) -> None:
        if self.embeddings is not None:
            np.save(self.cache_file, self.embeddings)

    def _create_text(self, paper: Dict[str, object]) -> str:
        parts: List[str] = []
        if paper.get("title"):
            parts.append(f"Title: {paper['title']}")
        if paper.get("authors"):
            authors = ", ".join(paper["authors"]) if isinstance(paper["authors"], list) else paper["authors"]
            parts.append(f"Authors: {authors}")
        if paper.get("year"):
            parts.append(f"Year: {paper['year']}")
        if paper.get("source"):
            parts.append(f"Source: {paper['source']}")
        if paper.get("abstract"):
            parts.append(f"Abstract: {paper['abstract']}")
        if paper.get("keywords"):
            keywords = (
                ", ".join(paper["keywords"]) if isinstance(paper["keywords"], list) else paper["keywords"]
            )
            parts.append(f"Keywords: {keywords}")
        return " ".join(parts)

    def _embed_openai(self, texts: Sequence[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        embeddings: List[Sequence[float]] = []
        batch_size = 100
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self.client.embeddings.create(input=batch, model=self.model_name)
            embeddings.extend(item.embedding for item in response.data)
        return np.array(embeddings)

    def _embed_local(self, texts: Sequence[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, show_progress_bar=len(texts) > 100)

    def compute_embeddings(self, force: bool = False) -> np.ndarray:
        if self.embeddings is not None and not force:
            return self.embeddings

        texts = [self._create_text(paper) for paper in self.papers]
        if self.model_type in {"openai", "siliconflow"}:
            self.embeddings = self._embed_openai(texts)
        else:
            self.embeddings = self._embed_local(texts)
        self._save_cache()
        return self.embeddings

    def search(
        self,
        *,
        examples: Optional[Iterable[Dict[str, object]]] = None,
        query: Optional[str] = None,
        top_k: int = 100,
    ) -> List[Dict[str, object]]:
        if self.embeddings is None:
            self.compute_embeddings()

        if examples:
            texts = []
            for example in examples:
                text = f"Title: {example.get('title', '')}"
                if example.get("abstract"):
                    text += f" Abstract: {example['abstract']}"
                texts.append(text)
            if self.model_type in {"openai", "siliconflow"}:
                query_emb = self._embed_openai(texts)
            else:
                query_emb = self._embed_local(texts)
            query_vector = np.mean(query_emb, axis=0).reshape(1, -1)
        elif query:
            if self.model_type in {"openai", "siliconflow"}:
                query_vector = self._embed_openai([query]).reshape(1, -1)
            else:
                query_vector = self._embed_local([query]).reshape(1, -1)
        else:
            raise ValueError("Provide either examples or query")

        similarities = cosine_similarity(query_vector, self.embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            {"paper": self.papers[idx], "similarity": float(similarities[idx])}
            for idx in top_indices
        ]

    def display(self, results: Sequence[Dict[str, object]], n: int = 10) -> None:
        print(f"\n{'=' * 80}")
        print(f"Top {len(results)} Results (showing {min(n, len(results))})")
        print(f"{'=' * 80}\n")
        for i, result in enumerate(results[:n], 1):
            paper = result["paper"]
            sim = result["similarity"]
            print(f"{i}. [{sim:.4f}] {paper.get('title', 'Untitled')}")
            number = paper.get("number") or paper.get("paper_id") or paper.get("eid") or "N/A"
            area = paper.get("primary_area") or paper.get("source") or paper.get("journal") or "N/A"
            link = (
                paper.get("forum_url")
                or paper.get("link")
                or paper.get("url")
                or paper.get("doi")
                or "N/A"
            )
            print(f"   #{number} | {area}")
            print(f"   {link}\n")

    def save(self, results: Sequence[Dict[str, object]], output_file: str) -> None:
        with open(output_file, "w", encoding="utf-8") as handle:
            json.dump({"model": self.model_name, "total": len(results), "results": list(results)}, handle, ensure_ascii=False, indent=2)
        print(f"Saved to {output_file}")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search similar research papers without writing code.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("papers_file", help="Path to the papers dataset (JSON/JSONL/CSV).")
    parser.add_argument("--format", dest="data_format", help="Explicitly specify the dataset format.")
    parser.add_argument(
        "--model",
        dest="model_type",
        choices=["local", "openai", "siliconflow"],
        default="openai",
        help="Embedding provider to use.",
    )
    parser.add_argument("--api-key", dest="api_key", help="Override the API key for hosted providers.")
    parser.add_argument("--base-url", dest="base_url", help="Custom base URL for OpenAI-compatible services.")
    parser.add_argument("--embedding-model", dest="embedding_model", help="Override the embedding model name.")
    parser.add_argument("--save", dest="output_file", help="Save results to the provided JSON file.")
    parser.add_argument("--top-k", dest="top_k", type=int, default=100, help="Number of results to retrieve.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    searcher = PaperSearcher(
        args.papers_file,
        model_type=args.model_type,
        api_key=args.api_key,
        base_url=args.base_url,
        data_format=args.data_format,
        embedding_model=args.embedding_model,
    )

    searcher.compute_embeddings()
    print("Enter a query (blank line to exit):")
    for line in iter(input, ""):
        query = line.strip()
        if not query:
            break
        results = searcher.search(query=query, top_k=args.top_k)
        searcher.display(results)
        if args.output_file:
            searcher.save(results, args.output_file)
        print("\nEnter another query (blank line to exit):")


if __name__ == "__main__":  # pragma: no cover
    main()

