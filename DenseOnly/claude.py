# -*- coding: utf-8 -*-
import os
import json
import time
import random
import math
import pickle
import hashlib
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
import requests
from dataclasses import dataclass
from abc import ABC, abstractmethod

# 配置參數
QUESTION = "我現在是碩二，準備要畢業了，我要怎麼跑畢業流程？"
TOP_K = 5

SOURCE_DB_JSON = "./data/school_rules.json"
REF_MAX_CHARS = int(os.getenv("REF_MAX_CHARS", "16000"))

# 是否對長文分塊（建議開啟，檢索更細緻）
USE_CHUNKING = os.getenv(
    "USE_CHUNKING", "true").lower() in ("1", "true", "yes")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Embedding 服務（自定 API）
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")  # 若需要驗證可填；否則可留空

# Embedding 快取
EMBEDDINGS_CACHE_PATH = "./embedding/embeddings_cache.pkl"

# Anthropic API
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_URL = os.getenv("ANTHROPIC_URL")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL")
MAX_TOKENS = 512
TEMPERATURE = 0.0

# =========================
# 系統與工具
# =========================
SYSTEM_PROMPT = (
    "你是嚴謹的法規/規章問答助手。"
    "請只依據「參考文獻」內容，以中文產生精簡、清楚、正確的回答；"
    "避免加入未在文獻中出現的推測或外部知識。"
    "若文獻未明確說明，直接回答「文獻未明確說明」。"
    "字數以 2–4 句為宜；若能指出條號，請自然置入（如：依第X條…）。"
)

# =========================
# 資料結構
# =========================


@dataclass
class Document:
    id: str
    title: str
    content: str
    metadata: Optional[Dict] = None


@dataclass
class RetrievalResult:
    document: Document
    score: float
    rank: int

# =========================
# Embedding 快取
# =========================


@dataclass
class EmbeddingCacheEntry:
    text_hash: str
    embedding: List[float]
    timestamp: float
    text_preview: str


class EmbeddingCache:
    def __init__(self, cache_file_path: Path):
        self.cache_file_path = Path(cache_file_path)
        self.cache: Dict[str, EmbeddingCacheEntry] = {}
        self.load_cache()

    def _get_text_hash(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def load_cache(self):
        try:
            if self.cache_file_path.exists():
                with open(self.cache_file_path, "rb") as f:
                    self.cache = pickle.load(f)
                print(f"已載入 {len(self.cache)} 個 embedding 緩存條目")
            else:
                print("未找到 embedding 緩存文件，將建立新的緩存")
        except Exception as e:
            print(f"載入 embedding 緩存失敗: {e}")
            self.cache = {}

    def save_cache(self):
        try:
            self.cache_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file_path, "wb") as f:
                pickle.dump(self.cache, f)
            print(f"已保存 {len(self.cache)} 個 embedding 緩存條目")
        except Exception as e:
            print(f"保存 embedding 緩存失敗: {e}")

    def get_embedding(self, text: str) -> Optional[List[float]]:
        key = self._get_text_hash(text)
        if key in self.cache:
            return self.cache[key].embedding
        return None

    def set_embedding(self, text: str, embedding: List[float]):
        import time
        key = self._get_text_hash(text)
        self.cache[key] = EmbeddingCacheEntry(
            text_hash=key,
            embedding=embedding,
            timestamp=time.time(),
            text_preview=text[:100] + "..." if len(text) > 100 else text
        )

    def get_cache_stats(self) -> Dict[str, Any]:
        return {
            "total_entries": len(self.cache),
            "cache_file_exists": self.cache_file_path.exists(),
            "cache_file_path": str(self.cache_file_path),
        }

# =========================
# Embedding Provider
# =========================


class EmbeddingProvider(ABC):
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        pass


class CustomEmbeddingProvider(EmbeddingProvider):
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
            vec = self.cache.get_embedding(text)
            if vec is not None:
                self.cache_hits += 1
                return vec

        self.api_calls += 1
        vec = self._call_api(text)

        if self.cache and vec:
            self.cache.set_embedding(text, vec)
        return vec

    def _call_api(self, text: str) -> List[float]:
        headers = {"accept": "application/json",
                   "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = {"inputs": text}
        try:
            resp = requests.post(self.api_url, headers=headers,
                                 json=data, timeout=self.timeout)
            resp.raise_for_status()
            result = resp.json()
            if isinstance(result, list) and result and isinstance(result[0], (list, float, int)):
                return result[0] if isinstance(result[0], list) else result
            if "embeddings" in result:
                emb = result["embeddings"]
                return emb[0] if (isinstance(emb, list) and emb and isinstance(emb[0], list)) else emb
            if "embedding" in result:
                return result["embedding"]
            for key in ("vectors", "data", "result"):
                if key in result:
                    vec = result[key]
                    return vec[0] if (isinstance(vec, list) and vec and isinstance(vec[0], list)) else vec
            raise ValueError(f"無法解析 embedding 響應格式: {result}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Embedding API 請求失敗: {e}")

    def get_stats(self) -> Dict[str, int]:
        return {
            "cache_hits": self.cache_hits,
            "api_calls": self.api_calls,
            "total_requests": self.cache_hits + self.api_calls
        }

# =========================
# Dense Retriever
# =========================


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        pass


class DenseRetriever(Retriever):
    def __init__(self, embedding_provider: EmbeddingProvider):
        self.embedding_provider = embedding_provider
        self.documents: List[Document] = []
        self.document_embeddings: List[List[float]] = []
        self._indexed = False

    def add_documents(self, documents: List[Document]):
        self.documents.extend(documents)
        self._indexed = False

    def build_index(self):
        if self._indexed:
            return
        print(f"正在為 {len(self.documents)} 個文檔構建向量索引...")
        self.document_embeddings = []
        for i, doc in enumerate(self.documents):
            try:
                text_to_embed = f"{doc.title}\n{doc.content[:1000]}"
                vec = self.embedding_provider.get_embedding(text_to_embed)
                self.document_embeddings.append(vec)
                if (i + 1) % 10 == 0:
                    print(f"已處理 {i + 1}/{len(self.documents)} 個文檔")
            except Exception as e:
                print(f"文檔 {doc.id} 建立索引失敗: {e}")
                self.document_embeddings.append([0.0] * 768)
        self._indexed = True
        print("向量索引構建完成")
        # 儲存 embedding 快取
        if hasattr(self.embedding_provider, "cache") and self.embedding_provider.cache:
            self.embedding_provider.cache.save_cache()

    @staticmethod
    def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
        if (not v1) or (not v2) or (len(v1) != len(v2)):
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        n1 = math.sqrt(sum(a * a for a in v1))
        n2 = math.sqrt(sum(b * b for b in v2))
        if n1 == 0 or n2 == 0:
            return 0.0
        return dot / (n1 * n2)

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        if not self._indexed:
            self.build_index()
        if not self.documents:
            return []
        qvec = self.embedding_provider.get_embedding(query)
        sims: List[Tuple[int, float]] = []
        for i, dvec in enumerate(self.document_embeddings):
            sims.append((i, self._cosine_similarity(qvec, dvec)))
        sims.sort(key=lambda x: x[1], reverse=True)
        top = sims[:max(1, top_k)]
        results: List[RetrievalResult] = []
        for rank, (idx, score) in enumerate(top, 1):
            results.append(RetrievalResult(
                document=self.documents[idx], score=score, rank=rank))
        return results

# =========================
# 輔助：載入文件 & 分塊
# =========================


def load_source_database(path: Path) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到出處資料庫檔案：{path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def documents_from_db(db: Dict) -> List[Document]:
    """
    將 {doc_key: value} 轉成 List[Document]
    - value 若為 dict(條文)，則依數字鍵排序後合併成全文
    - value 若為 str，則直接當全文
    """
    docs: List[Document] = []
    for doc_key, val in db.items():
        if isinstance(val, dict):
            def _key_order(k: str):
                return int(k) if str(k).isdigit() else float("inf")
            parts: List[str] = []
            for k in sorted(val.keys(), key=_key_order):
                parts.append(f"{val[k]}")
            full = "\n".join(parts)
        else:
            full = str(val)
        docs.append(Document(id=str(doc_key), title=str(doc_key), content=full, metadata={
                    "original_structure": val if isinstance(val, dict) else None}))
    return docs


def create_chunked_documents(documents: List[Document], chunk_size: int = 800, chunk_overlap: int = 100) -> List[Document]:
    chunked: List[Document] = []
    for doc in documents:
        content = doc.content
        if len(content) <= chunk_size:
            chunked.append(doc)
            continue
        start = 0
        idx = 0
        while start < len(content):
            end = min(start + chunk_size, len(content))
            chunk_text = content[start:end]
            chunked.append(
                Document(
                    id=f"{doc.id}_chunk_{idx}",
                    title=f"{doc.title} (第{idx+1}段)",
                    content=chunk_text,
                    metadata={"parent_doc_id": doc.id, "chunk_index": idx,
                              "is_chunk": True, **(doc.metadata or {})}
                )
            )
            idx += 1
            if end >= len(content):
                break
            start = max(end - chunk_overlap, start + 1)
    return chunked

# =========================
# Prompt 組裝 & Anthropic 呼叫
# =========================


def build_user_prompt(question: str, references: List[RetrievalResult]) -> str:
    if references:
        ref_texts = []
        total = 0
        for r in references:
            piece = f"【{r.document.title}】\n{r.document.content}"
            # 依 REF_MAX_CHARS 截斷
            usable = REF_MAX_CHARS - total
            if usable <= 0:
                break
            if len(piece) > usable:
                piece = piece[:usable] + "…（截斷）"
            ref_texts.append(piece)
            total += len(piece)
        ref_content = "\n\n".join(ref_texts) if ref_texts else "（無）"
    else:
        ref_content = "（無）"

    return f"""參考文獻：
{ref_content}

問題：
{question.strip()}

請以中文回答，務必精簡清楚，且只根據參考文獻作答。不要加上多餘前後綴。"""


def anthropic_call(user_prompt: str, system_prompt: str) -> str:
    if not ANTHROPIC_API_KEY:
        raise EnvironmentError("請先設定環境變數 ANTHROPIC_API_KEY")

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }

    last_err = None
    for attempt in range(3):
        try:
            resp = requests.post(
                ANTHROPIC_URL, headers=headers, json=payload, timeout=90)
            if resp.status_code == 429:
                raise RuntimeError(f"Rate limited: {resp.text}")
            resp.raise_for_status()
            data = resp.json()
            texts = [b.get("text", "") for b in data.get(
                "content", []) if b.get("type") == "text"]
            return (texts[0].strip() if texts else "").strip()
        except Exception as e:
            last_err = e
            time.sleep(min(1.0 * (2 ** attempt) + random.uniform(0, 0.5), 3.0))
    raise RuntimeError(f"Anthropic API failed after retries: {last_err}")

# =========================
# 主流程
# =========================


def main():
    print("處理題目 1...")

    # 1) 載入資料庫，轉 Document；可選分塊
    db = load_source_database(SOURCE_DB_JSON)
    print(f"載入了 {len(db)} 個出處（doc_key）")

    docs = documents_from_db(db)
    print(f"原始文檔：{len(docs)}")
    if USE_CHUNKING:
        print("正在進行文檔分塊處理...")
        docs = create_chunked_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"分塊後共有 {len(docs)} 個文檔片段")

    # 2) 初始化 Embedding 快取與 Provider、Retriever
    cache = EmbeddingCache(EMBEDDINGS_CACHE_PATH)
    print(f"緩存統計: {cache.get_cache_stats()}")

    embedding_provider = CustomEmbeddingProvider(
        api_url=EMBEDDING_API_URL,
        api_key=EMBEDDING_API_KEY,
        cache=cache
    )
    retriever = DenseRetriever(embedding_provider)
    retriever.add_documents(docs)

    # 3) 檢索
    print(f"正在檢索與問題相關的文檔 (top_k={TOP_K})...")
    results = retriever.retrieve(QUESTION, top_k=TOP_K)

    # 4) 組 prompt 並呼叫 Claude
    user_prompt = build_user_prompt(QUESTION, results)
    try:
        answer = anthropic_call(user_prompt, SYSTEM_PROMPT)
    except Exception as e:
        print(f"發生錯誤：{e}")
        return

    # 5) 輸出
    print("\n題目 1 完成 - 答案如下：\n")
    print(answer)

    # 額外：列出檢索命中與快取統計
    print("\n檢索到的參考文獻（按相似度排序）:")
    for r in results:
        print(f"{r.rank}. {r.document.title} (相似度: {r.score:.4f})")

    print("\nEmbedding 統計：")
    stats = embedding_provider.get_stats()
    print(f"  緩存命中: {stats['cache_hits']}")
    print(f"  API 調用: {stats['api_calls']}")
    print(f"  總請求: {stats['total_requests']}")
    if stats["total_requests"] > 0:
        hit_rate = stats["cache_hits"] / stats["total_requests"] * 100
        print(f"  緩存命中率: {hit_rate:.1f}%")

    print("\n完成！")


if __name__ == "__main__":
    main()
