"""Streamlit application for visually exploring paper embeddings."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

from config_utils import get_setting
from search import PaperSearcher


st.set_page_config(page_title="Paper Search Assistant", layout="wide")
st.title("📄 Paper Search Assistant")
st.markdown(
    """
    Upload your research paper dataset, pick an embedding provider, and enter a
    short description of the papers you're looking for. The app will compute
    embeddings, rank the most relevant items, and let you download the results
    for later review.
    """
)


def _persist_uploaded_file(file) -> Optional[Path]:
    if file is None:
        return None

    suffix = Path(file.name).suffix or ".json"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(file.getbuffer())
        temp_path = Path(handle.name)
    st.session_state["uploaded_name"] = file.name
    st.session_state["uploaded_suffix"] = suffix.lstrip(".")
    st.session_state["uploaded_path"] = str(temp_path)
    return temp_path


def _parse_mapping(raw_text: str) -> Dict[str, Any]:
    if not raw_text.strip():
        return {}
    try:
        mapping = json.loads(raw_text)
        if not isinstance(mapping, dict):
            raise ValueError("Mapping must be a JSON object.")
        return mapping
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON mapping: {exc}") from exc


def _load_searcher(
    data_path: Path,
    data_format: Optional[str],
    model_type: str,
    api_key: Optional[str],
    base_url: Optional[str],
    embedding_model: Optional[str],
    csv_mapping: Dict[str, Any],
):
    return PaperSearcher(
        str(data_path),
        model_type=model_type,
        api_key=api_key or None,
        base_url=base_url or None,
        data_format=data_format,
        csv_mapping=csv_mapping,
        embedding_model=embedding_model or None,
    )


st.header("1. 数据集")
uploaded_file = st.file_uploader(
    "上传包含论文的 JSON、JSONL 或 CSV 文件", type=["json", "jsonl", "csv"]
)

data_path: Optional[Path] = None
data_format: Optional[str] = None
if uploaded_file is not None:
    data_path = _persist_uploaded_file(uploaded_file)
elif "uploaded_path" in st.session_state:
    stored_path = Path(st.session_state["uploaded_path"])
    if stored_path.exists():
        data_path = stored_path
        st.info(
            f"使用之前上传的文件：{st.session_state.get('uploaded_name', stored_path.name)}"
            " — 如需更换请重新上传。"
        )

if data_path is not None:
    data_format = Path(data_path).suffix.lstrip(".").lower()
elif "uploaded_suffix" in st.session_state:
    data_format = st.session_state["uploaded_suffix"].lower()

default_mapping_hint = (
    "{\n  \"title\": [\"文献标题\"],\n  \"abstract\": [\"摘要\"],\n  "
    "\"authors\": [\"作者\"],\n  \"keywords\": [\"作者关键字\"],\n  "
    "\"year\": [\"年份\"]\n}"
)

with st.expander("高级：自定义 CSV 字段映射"):
    st.markdown(
        """
        如果 CSV 列名与默认的 Scopus 或通用字段不同，可以在此粘贴 JSON
        格式的字段映射。例如：
        """
    )
    st.code(default_mapping_hint, language="json")
    mapping_text = st.text_area("自定义映射", value="", height=160)
    try:
        custom_mapping = _parse_mapping(mapping_text) if mapping_text else {}
    except ValueError as exc:
        st.error(str(exc))
        custom_mapping = {}


st.header("2. 向量模型设置")
cols = st.columns(3)
with cols[0]:
    model_type = st.selectbox(
        "嵌入服务提供商",
        options=["local", "openai", "siliconflow"],
        format_func=lambda x: {
            "local": "本地（离线）",
            "openai": "OpenAI",
            "siliconflow": "硅基流动",
        }[x],
    )

with cols[1]:
    embedding_model = st.text_input(
        "嵌入模型（可选）",
        placeholder={
            "local": "例如 all-MiniLM-L6-v2",
            "openai": "例如 text-embedding-3-large",
            "siliconflow": "例如 BAAI/bge-large-zh-v1.5",
        }[model_type],
    )

with cols[2]:
    base_url = st.text_input(
        "自定义 API Base URL",
        value="https://api.siliconflow.cn/v1" if model_type == "siliconflow" else "",
        help="如使用自建 OpenAI 兼容服务，可在此覆盖默认地址。",
    )

if model_type == "openai":
    configured_key = get_setting("OPENAI_API_KEY")
elif model_type == "siliconflow":
    configured_key = get_setting("SILICONFLOW_API_KEY")
else:
    configured_key = None

api_key = st.text_input(
    "API Key（若选择线上服务）",
    type="password",
    value=configured_key or "",
    placeholder="请在 .env 或 config.json 中配置对应的 API Key",
    key=f"api_key_{model_type}",
)

recompute = st.checkbox(
    "重新计算嵌入", value=False, help="勾选后会忽略缓存并重新生成嵌入向量。"
)


st.header("3. 检索配置")
query = st.text_area("请输入检索需求描述", placeholder="例如：社交机器人在零售环境中的应用")
col_search = st.columns(2)
with col_search[0]:
    top_k = st.slider("检索结果数量", min_value=1, max_value=200, value=10)
with col_search[1]:
    show_k = st.slider("界面展示条数", min_value=1, max_value=50, value=10)


run_search = st.button("🔍 开始搜索", type="primary")

results_state: Optional[Dict[str, Any]] = st.session_state.get("results")

if run_search:
    if data_path is None:
        st.error("请先上传数据文件。")
    elif not query.strip():
        st.error("请输入检索描述。")
    else:
        with st.spinner("正在计算嵌入并检索相似论文..."):
            try:
                searcher = _load_searcher(
                    data_path,
                    data_format,
                    model_type,
                    api_key if api_key else None,
                    base_url or None,
                    embedding_model or None,
                    custom_mapping,
                )
                searcher.compute_embeddings(force=recompute)
                items = searcher.search(query=query, top_k=top_k)
                st.session_state["results"] = {
                    "items": items,
                    "model_name": searcher.model_name,
                    "provider": model_type,
                }
                results_state = st.session_state["results"]
            except Exception as exc:
                st.error(f"检索失败：{exc}")
                results_state = None


if results_state and results_state.get("items"):
    st.header("🔎 检索结果")
    items = results_state["items"]
    for idx, item in enumerate(items[:show_k], start=1):
        paper = item["paper"]
        similarity = item["similarity"]
        title = paper.get("title", "未命名论文")
        st.markdown(f"**{idx}. [{similarity:.4f}] {title}**")

        meta_cols = st.columns(3)
        meta_cols[0].metric("年份", paper.get("year", "-"))
        meta_cols[1].metric("来源", paper.get("source") or paper.get("journal", "-"))
        meta_cols[2].metric("编号", paper.get("number") or paper.get("paper_id") or paper.get("eid", "-"))

        authors = paper.get("authors")
        if isinstance(authors, list):
            authors_display = ", ".join(authors)
        else:
            authors_display = authors or "-"

        st.caption(f"作者：{authors_display}")

        if paper.get("abstract"):
            with st.expander("摘要"):
                st.write(paper["abstract"])

        link = paper.get("forum_url") or paper.get("link") or paper.get("url") or paper.get("doi")
        if link:
            st.markdown(f"[查看原文]({link})")

        st.divider()

    download_payload = json.dumps(
        {
            "model": results_state.get("model_name"),
            "provider": results_state.get("provider"),
            "results": items,
        },
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")
    st.download_button(
        "💾 下载完整结果（JSON）",
        download_payload,
        file_name="paper_search_results.json",
        mime="application/json",
    )

