import os
import json
import math
import time
import random
import pickle
import hashlib
from typing import Dict, List, Optional, Any, Tuple

import requests
import google.generativeai as genai

# =========================
# 必填：題目
# =========================
QUESTION = """我要畢業，要怎麼跑流程？"""

# =========================
# 檔案與模型設定
# =========================
SOURCE_DB_JSON = "./data/school_rules.json"
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
REF_MAX_CHARS = int(os.getenv("REF_MAX_CHARS", "16000"))

TOP_K = 5
USE_CHUNKING = os.getenv(
    "USE_CHUNKING", "true").lower() in ("1", "true", "yes")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")
EMBEDDINGS_CACHE_PATH = "./embedding/embeddings_cache.pkl"

# =========================
# 資料結構
# =========================


class Document:
    def __init__(self, id: str, title: str, content: str, metadata: Optional[Dict] = None):
        self.id = id
        self.title = title
        self.content = content
        self.metadata = metadata or {}


class RetrievalResult:
    def __init__(self, document: Document, score: float, rank: int):
        self.document = document
        self.score = score
        self.rank = rank

# =========================
# Embedding 緩存
# =========================


class EmbeddingCacheEntry:
    def __init__(self, text_hash: str, embedding: List[float], timestamp: float, text_preview: str):
        self.text_hash = text_hash
        self.embedding = embedding
        self.timestamp = timestamp
        self.text_preview = text_preview


class EmbeddingCache:
    def __init__(self, cache_file_path: str):
        self.cache_file_path = cache_file_path
        self.cache: Dict[str, EmbeddingCacheEntry] = {}
        self.load_cache()

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def load_cache(self):
        try:
            if os.path.exists(self.cache_file_path):
                with open(self.cache_file_path, "rb") as f:
                    self.cache = pickle.load(f)
                print(f"已載入 {len(self.cache)} 個 embedding 緩存條目")
            else:
                print("未找到 embedding 緩存文件，將建立新的緩存")
        except Exception as e:
            print(f"載入 embedding 緩存失敗: {e}")
            self.cache = {}

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.cache_file_path), exist_ok=True)
            with open(self.cache_file_path, "wb") as f:
                pickle.dump(self.cache, f)
            print(f"已保存 {len(self.cache)} 個 embedding 緩存條目")
        except Exception as e:
            print(f"保存 embedding 緩存失敗: {e}")

    def get(self, text: str) -> Optional[List[float]]:
        key = self._hash(text)
        if key in self.cache:
            return self.cache[key].embedding
        return None

    def set(self, text: str, embedding: List[float]):
        key = self._hash(text)
        self.cache[key] = EmbeddingCacheEntry(
            text_hash=key,
            embedding=embedding,
            timestamp=time.time(),
            text_preview=text[:100] + "..." if len(text) > 100 else text
        )

    def stats(self) -> Dict[str, Any]:
        return {
            "total_entries": len(self.cache),
            "cache_file_exists": os.path.exists(self.cache_file_path),
            "cache_file_path": self.cache_file_path,
        }

# =========================
# Embedding Provider
# =========================


class EmbeddingProvider:
    def __init__(self, api_url: str, api_key: Optional[str] = None, timeout: int = 30, cache: Optional[EmbeddingCache] = None):
        if not api_url:
            raise EnvironmentError("請設定 EMBEDDING_API_URL")
        self.api_url = api_url
        self.api_key = api_key
        self.timeout = timeout
        self.cache = cache
        self.cache_hits = 0
        self.api_calls = 0

    def get_embedding(self, text: str) -> List[float]:
        if self.cache:
            v = self.cache.get(text)
            if v is not None:
                self.cache_hits += 1
                return v

        self.api_calls += 1
        v = self._call_api(text)
        if self.cache and v:
            self.cache.set(text, v)
        return v

    def _call_api(self, text: str) -> List[float]:
        headers = {"accept": "application/json",
                   "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = {"inputs": text}
        try:
            r = requests.post(self.api_url, headers=headers,
                              json=data, timeout=self.timeout)
            r.raise_for_status()
            res = r.json()
            if isinstance(res, list) and res:
                return res[0] if isinstance(res[0], list) else res
            if "embeddings" in res:
                e = res["embeddings"]
                return e[0] if (isinstance(e, list) and e and isinstance(e[0], list)) else e
            if "embedding" in res:
                return res["embedding"]
            for k in ("vectors", "data", "result"):
                if k in res:
                    v = res[k]
                    return v[0] if (isinstance(v, list) and v and isinstance(v[0], list)) else v
            raise ValueError(f"無法解析 embedding 響應格式: {res}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Embedding API 請求失敗: {e}")

    def stats(self) -> Dict[str, int]:
        return {
            "cache_hits": self.cache_hits,
            "api_calls": self.api_calls,
            "total_requests": self.cache_hits + self.api_calls,
        }

# =========================
# Dense Retriever
# =========================


class DenseRetriever:
    def __init__(self, embedding_provider: EmbeddingProvider):
        self.embedding_provider = embedding_provider
        self.documents: List[Document] = []
        self.doc_embeddings: List[List[float]] = []
        self._indexed = False

    def add_documents(self, docs: List[Document]):
        self.documents.extend(docs)
        self._indexed = False

    def _cosine(self, a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def build_index(self):
        if self._indexed:
            return
        print(f"正在為 {len(self.documents)} 個文檔構建向量索引...")
        self.doc_embeddings = []
        for i, d in enumerate(self.documents):
            try:
                text = f"{d.title}\n{d.content[:1000]}"
                vec = self.embedding_provider.get_embedding(text)
                self.doc_embeddings.append(vec)
                if (i + 1) % 10 == 0:
                    print(f"已處理 {i + 1}/{len(self.documents)} 個文檔")
            except Exception as e:
                print(f"文檔 {d.id} 建索引失敗: {e}")
                self.doc_embeddings.append([0.0] * 768)
        self._indexed = True
        print("向量索引構建完成")
        if self.embedding_provider.cache:
            self.embedding_provider.cache.save()

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        if not self._indexed:
            self.build_index()
        if not self.documents:
            return []
        qv = self.embedding_provider.get_embedding(query)
        sims: List[Tuple[int, float]] = []
        for i, dv in enumerate(self.doc_embeddings):
            sims.append((i, self._cosine(qv, dv)))
        sims.sort(key=lambda x: x[1], reverse=True)
        sims = sims[:max(1, top_k)]
        return [RetrievalResult(self.documents[idx], score, rank) for rank, (idx, score) in enumerate(sims, 1)]

# =========================
# 輔助：載入 DB 與分塊
# =========================


def load_db(path: str) -> Dict:
    if not os.path.exists(path):
        raise RuntimeError(f"找不到資料庫 JSON：{path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def documents_from_db(db: Dict) -> List[Document]:
    docs: List[Document] = []
    for doc_key, val in db.items():
        if isinstance(val, dict):
            def _k(k: str):
                return int(k) if str(k).isdigit() else float("inf")
            parts = [f"{val[k]}" for k in sorted(val.keys(), key=_k)]
            full = "\n".join(parts)
        else:
            full = str(val)
        docs.append(Document(id=str(doc_key), title=str(doc_key), content=full, metadata={
                    "original_structure": val if isinstance(val, dict) else None}))
    return docs


def chunk_documents(docs: List[Document], size: int = 800, overlap: int = 100) -> List[Document]:
    out: List[Document] = []
    for d in docs:
        text = d.content
        if len(text) <= size:
            out.append(d)
            continue
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + size, len(text))
            chunk = text[start:end]
            out.append(Document(
                id=f"{d.id}_chunk_{idx}",
                title=f"{d.title} (第{idx+1}段)",
                content=chunk,
                metadata={"parent_doc_id": d.id, "chunk_index": idx,
                          "is_chunk": True, **(d.metadata or {})}
            ))
            idx += 1
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return out

# =========================
# Prompt & Gemini 呼叫
# =========================


def load_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        key_file = os.path.join(os.path.dirname(__file__), ".google_api_key")
        if os.path.exists(key_file):
            api_key = open(key_file, "r", encoding="utf-8").read().strip()
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY 未設定，且找不到 .google_api_key")
    return api_key


def build_prompt(question: str, results: List[RetrievalResult]) -> str:
    pieces: List[str] = []
    total = 0
    for r in results:
        block = f"【{r.document.title}】\n{r.document.content}"
        remain = REF_MAX_CHARS - total
        if remain <= 0:
            break
        if len(block) > remain:
            block = block[:remain] + "…（截斷）"
        pieces.append(block)
        total += len(block)
    ref = "\n\n".join(pieces) if pieces else "（無）"
    return (
        "根據以下參考文獻，以中文精簡回答使用者問題；"
        "若文獻未提到就說「文獻未明確說明」。\n"
        f"{ref}\n問題：{question.strip()}"
    )


def main():
    api_key = load_api_key()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)

    db = load_db(SOURCE_DB_JSON)
    print(f"載入了 {len(db)} 個出處（doc_key）")

    docs = documents_from_db(db)
    print(f"原始文檔：{len(docs)}")
    if USE_CHUNKING:
        print("正在進行文檔分塊處理...")
        docs = chunk_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"分塊後共有 {len(docs)} 個文檔片段")

    cache = EmbeddingCache(EMBEDDINGS_CACHE_PATH)
    print(f"緩存統計: {cache.stats()}")
    provider = EmbeddingProvider(
        api_url=EMBEDDING_API_URL, api_key=EMBEDDING_API_KEY, cache=cache)
    retriever = DenseRetriever(provider)
    retriever.add_documents(docs)

    print(f"正在檢索與問題相關的文檔 (top_k={TOP_K})...")
    results = retriever.retrieve(QUESTION, top_k=TOP_K)
    prompt = build_prompt(QUESTION, results)
    resp = model.generate_content(prompt)
    answer = (resp.text or "").strip()

    print("\n題目 1 完成 - 答案如下：\n")
    print(answer)

    print("\n檢索到的參考文獻（按相似度排序）:")
    for r in results:
        print(f"{r.rank}. {r.document.title} (相似度: {r.score:.4f})")

    stats = provider.stats()
    print("\nEmbedding 統計：")
    print(f"  緩存命中: {stats['cache_hits']}")
    print(f"  API 調用: {stats['api_calls']}")
    print(f"  總請求: {stats['total_requests']}")
    if stats["total_requests"] > 0:
        rate = stats["cache_hits"] / stats["total_requests"] * 100
        print(f"  緩存命中率: {rate:.1f}%")


# =========================
# 主流程
# =========================
if __name__ == "__main__":
    main()
