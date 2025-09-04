#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
School Regulation Q&A System with Dense Retriever
- 加入 embedding 磁碟快取：./embedding/embeddings_cache.pkl
"""

import json
import math
import os
from typing import Dict, List, Tuple, Optional, Any
import requests
from dataclasses import dataclass
from abc import ABC, abstractmethod
from dotenv import load_dotenv

# 新增：快取所需
import pickle
import hashlib
from pathlib import Path

# 讀取環境變數（要在使用前呼叫）
load_dotenv()

QUESTION = "我現在是碩二，準備要畢業了，我要怎麼跑畢業流程？"

# ========== 外部服務設定 ==========
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL")
EMBEDDING_API_KEY = None  # 如果需要的話

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL")
GPT_OSS_API_KEY = os.getenv("GPT_OSS_API_KEY")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

DOCUMENTS_JSON_PATH = "./data/school_rules.json"

# ========== 檢索配置 ==========
TOP_K = 50         # 初始檢索的文檔數量 (reranker 前)
FINAL_TOP_K = 5    # 最終用於 LLM 的文檔數量 (reranker 後)
MAX_REF_LENGTH = 8000  # 參考文獻最大字數限制

# ========== Reranker 配置 ==========
USE_RERANKER = True  # 是否使用 reranker
RERANKER_MODEL = os.getenv("RERANKER_MODEL")
RERANKER_DEVICE = 'cuda'  # None 為自動選擇，也可以指定 'cuda', 'cpu', 'mps'

# ========== 分塊設定 ==========
USE_CHUNKING = True
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# ========== Embedding 快取路徑 ==========
EMBEDDING_CACHE_PATH = "./embedding/embeddings_cache.pkl"

# 為 reranker 添加必要的 imports
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("警告: transformers 和 torch 未安裝，reranker 功能將不可用")
    print("請執行: pip install transformers torch")

try:
    from FlagEmbedding import FlagReranker
    FLAG_EMBEDDING_AVAILABLE = True
except ImportError:
    FLAG_EMBEDDING_AVAILABLE = False
    print("警告: FlagEmbedding 未安裝，將使用 transformers 實現 reranker")
    print("可選擇安裝: pip install FlagEmbedding")


@dataclass
class Document:
    """文檔數據結構"""
    id: str
    title: str
    content: str
    metadata: Optional[Dict] = None


@dataclass
class RetrievalResult:
    """檢索結果數據結構"""
    document: Document
    score: float
    rank: int


class EmbeddingProvider(ABC):
    """Embedding 提供者抽象基類"""

    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        pass


class CustomEmbeddingProvider(EmbeddingProvider):
    """使用自定義 API 的 Embedding 提供者（含磁碟快取）"""

    def __init__(self, api_url: str, api_key: Optional[str] = None, timeout: int = 30):
        self.api_url = api_url
        self.api_key = api_key
        self.timeout = timeout

        # 準備快取
        self.cache_path = Path(EMBEDDING_CACHE_PATH)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, List[float]] = self._load_cache()

        # 觀察到第一次成功向量的維度，用於出錯時的占位
        self._observed_dim: Optional[int] = None

    def _hash_text(self, text: str) -> str:
        """以內容為 key（對長文本也穩定），降低無關差異影響"""
        h = hashlib.sha256()
        normalized = text.replace("\r\n", "\n").strip()
        h.update(normalized.encode("utf-8"))
        return h.hexdigest()

    def _load_cache(self) -> Dict[str, List[float]]:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "rb") as f:
                    data = pickle.load(f)
                    if isinstance(data, dict):
                        # 嘗試讀取第一個向量維度
                        for v in data.values():
                            if isinstance(v, list):
                                self._observed_dim = len(v)
                                break
                        return data
            except Exception as e:
                print(f"警告：讀取 embedding 快取失敗，將重新建立（原因：{e}）")
        return {}

    def _save_cache(self) -> None:
        try:
            with open(self.cache_path, "wb") as f:
                pickle.dump(self._cache, f)
        except Exception as e:
            print(f"警告：寫入 embedding 快取失敗：{e}")

    def _post_and_parse(self, text: str) -> List[float]:
        """呼叫 API 並解析常見格式"""
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'

        data = {'inputs': text}

        response = requests.post(
            self.api_url,
            headers=headers,
            json=data,
            timeout=self.timeout
        )
        response.raise_for_status()
        result = response.json()

        if isinstance(result, list) and len(result) > 0:
            embedding = result[0] if isinstance(result[0], list) else result
        elif "embeddings" in result:
            embeddings = result["embeddings"]
            embedding = embeddings[0] if (isinstance(
                embeddings, list) and embeddings and isinstance(embeddings[0], list)) else embeddings
        elif "embedding" in result:
            embedding = result["embedding"]
        else:
            for k in ["vectors", "data", "result"]:
                if k in result:
                    vec = result[k]
                    if isinstance(vec, list) and vec and isinstance(vec[0], list):
                        embedding = vec[0]
                    else:
                        embedding = vec
                    break
            else:
                raise ValueError(f"無法解析 embedding 響應格式: {result}")

        if not isinstance(embedding, list) or not all(isinstance(x, (int, float)) for x in embedding):
            raise ValueError(f"embedding 格式非數值陣列: {type(embedding)}")

        # 記錄維度
        if self._observed_dim is None:
            self._observed_dim = len(embedding)

        return embedding

    def get_embedding(self, text: str) -> List[float]:
        """獲取文本的向量表示（含磁碟快取）"""
        if not self.api_url:
            raise RuntimeError("未設定 EMBEDDING_API_URL")

        key = f"v1:{self._hash_text(text)}"
        if key in self._cache:
            return self._cache[key]

        try:
            embedding = self._post_and_parse(text)
            self._cache[key] = embedding
            self._save_cache()
            return embedding

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Embedding API 請求失敗: {e}")


class Retriever(ABC):
    """檢索器抽象基類"""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        pass


class DenseRetriever(Retriever):
    """Dense 檢索器實現"""

    def __init__(self, embedding_provider: EmbeddingProvider):
        self.embedding_provider = embedding_provider
        self.documents: List[Document] = []
        self.document_embeddings: List[List[float]] = []
        self._indexed = False
        self._embed_dim: Optional[int] = None  # 觀察到的向量維度

    def add_documents(self, documents: List[Document]):
        """添加文檔到檢索器"""
        self.documents.extend(documents)
        self._indexed = False

    def build_index(self):
        """構建向量索引"""
        if self._indexed:
            return

        print(f"正在為 {len(self.documents)} 個文檔構建向量索引...")
        self.document_embeddings = []

        for i, doc in enumerate(self.documents):
            try:
                # 可選：僅對標題或標題+內容
                text_to_embed = f"{doc.title}\n{doc.content[:1000]}"
                embedding = self.embedding_provider.get_embedding(
                    text_to_embed)
                self.document_embeddings.append(embedding)
                if self._embed_dim is None:
                    self._embed_dim = len(embedding)

                if (i + 1) % 10 == 0:
                    print(f"已處理 {i + 1}/{len(self.documents)} 個文檔")

            except Exception as e:
                print(f"文檔 {doc.id} 建立索引失敗: {e}")
                dim = self._embed_dim or 768  # 若未知，先假設 768
                self.document_embeddings.append([0.0] * dim)

        self._indexed = True
        print("向量索引構建完成")

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """計算餘弦相似度"""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """檢索相關文檔"""
        if not self._indexed:
            self.build_index()

        if not self.documents:
            return []

        # 獲取查詢向量
        query_embedding = self.embedding_provider.get_embedding(query)
        if self._embed_dim is None:
            self._embed_dim = len(query_embedding)

        # 計算相似度
        similarities = []
        for i, doc_embedding in enumerate(self.document_embeddings):
            # 維度不一致時跳過或補齊（保守處理：回傳 0）
            if len(doc_embedding) != len(query_embedding):
                similarity = 0.0
            else:
                similarity = self.cosine_similarity(
                    query_embedding, doc_embedding)
            similarities.append((i, similarity))

        # 排序並獲取 top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:top_k]

        # 構建結果
        results = []
        for rank, (doc_idx, score) in enumerate(top_results, 1):
            result = RetrievalResult(
                document=self.documents[doc_idx],
                score=score,
                rank=rank
            )
            results.append(result)

        return results


class Reranker(ABC):
    """重排序器抽象基類"""

    @abstractmethod
    def rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        pass


class BGERerankerV2M3(Reranker):
    """BAAI/bge-reranker-v2-m3 重排序器實現"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3",
                 device: Optional[str] = None,
                 max_length: int = 512,
                 batch_size: int = 16):
        """
        初始化 BGE Reranker

        Args:
            model_name: 模型名稱
            device: 設備 ('cuda', 'cpu', 'mps' 或 None 自動選擇)
            max_length: 最大序列長度
            batch_size: 批次大小
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size

        # 自動選擇設備
        if device is None:
            if 'torch' in globals() and TRANSFORMERS_AVAILABLE and torch.cuda.is_available():
                device = 'cuda'
            elif 'torch' in globals() and TRANSFORMERS_AVAILABLE and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = device

        # 優先使用 FlagEmbedding，如果不可用則使用 transformers
        self.use_flag_embedding = FLAG_EMBEDDING_AVAILABLE
        self._load_model()

    def _load_model(self):
        """載入模型"""
        try:
            if self.use_flag_embedding:
                print(f"使用 FlagEmbedding 載入 reranker: {self.model_name}")
                self.reranker = FlagReranker(
                    self.model_name,
                    use_fp16=self.device != 'cpu'
                )
            else:
                if not TRANSFORMERS_AVAILABLE:
                    raise ImportError("transformers 和 torch 未安裝")

                print(f"使用 Transformers 載入 reranker: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name)
                self.model.to(self.device)
                self.model.eval()

            print(f"Reranker 載入成功 (設備: {self.device})")

        except Exception as e:
            raise RuntimeError(f"載入 reranker 模型失敗: {e}")

    def _compute_scores_flag_embedding(self, query: str, texts: List[str]) -> List[float]:
        """使用 FlagEmbedding 計算分數"""
        pairs = [[query, text] for text in texts]
        scores = self.reranker.compute_score(pairs, normalize=True)
        return scores if isinstance(scores, list) else [scores]

    def _compute_scores_transformers(self, query: str, texts: List[str]) -> List[float]:
        """使用 Transformers 計算分數"""
        scores = []

        # 分批處理
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_pairs = [f"{query} [SEP] {text}" for text in batch_texts]

            # Tokenize
            inputs = self.tokenizer(
                batch_pairs,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors='pt'
            ).to(self.device)

            # 推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_scores = torch.nn.functional.sigmoid(
                    outputs.logits.squeeze(-1))
                scores.extend(batch_scores.cpu().tolist())

        return scores

    def rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """重新排序檢索結果"""
        if not results:
            return results

        print(f"正在使用 BGE Reranker 重新排序 {len(results)} 個結果...")

        # 提取文本內容
        texts = []
        for result in results:
            text = f"{result.document.title}\n{result.document.content}"
            if len(text) > 2000:  # 保守估計，避免超過 token 限制
                text = text[:2000] + "..."
            texts.append(text)

        try:
            # 計算重排序分數
            if self.use_flag_embedding:
                rerank_scores = self._compute_scores_flag_embedding(
                    query, texts)
            else:
                rerank_scores = self._compute_scores_transformers(query, texts)

            # 更新結果的分數和排名
            reranked_results = []
            for i, (result, score) in enumerate(zip(results, rerank_scores)):
                new_result = RetrievalResult(
                    document=result.document,
                    score=float(score),  # 使用 rerank 分數
                    rank=i + 1
                )
                reranked_results.append(new_result)

            # 按新分數排序
            reranked_results.sort(key=lambda x: x.score, reverse=True)

            # 更新最終排名
            for i, result in enumerate(reranked_results):
                result.rank = i + 1

            print(
                f"重排序完成，分數範圍: {min(rerank_scores):.4f} - {max(rerank_scores):.4f}")
            return reranked_results

        except Exception as e:
            print(f"重排序過程出錯: {e}，返回原始結果")
            return results


class HuggingFaceReranker(Reranker):
    """通用的 Hugging Face Reranker (備用實現)"""

    def __init__(self, model_name: str, device: Optional[str] = None):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers 和 torch 未安裝")

        self.model_name = model_name
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name)
        self.model.to(self.device)
        self.model.eval()

    def rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """重新排序檢索結果"""
        if not results:
            return results

        texts = [
            f"{r.document.title} {r.document.content[:500]}" for r in results]
        pairs = [f"{query} [SEP] {text}" for text in texts]

        inputs = self.tokenizer(pairs, truncation=True, padding=True,
                                max_length=512, return_tensors='pt').to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = torch.nn.functional.sigmoid(outputs.logits.squeeze(-1))

        for i, (result, score) in enumerate(zip(results, scores.cpu().tolist())):
            result.score = score

        results.sort(key=lambda x: x.score, reverse=True)
        for i, result in enumerate(results):
            result.rank = i + 1

        return results


class LLMProvider(ABC):
    """LLM 提供者抽象基類"""

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]]) -> str:
        pass


class OllamaProvider(LLMProvider):
    """Ollama LLM 提供者"""

    def __init__(self, api_url: str, model: str, api_key: Optional[str] = None,
                 temperature: float = 0.2, timeout: int = 180):
        self.api_url = api_url
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """生成回答"""
        if not self.api_url or not self.model:
            raise RuntimeError("未設定 GPT_OSS_API_URL 或 GPT_OSS_MODEL")

        headers = {
            'Content-Type': 'application/json',
            'accept': 'application/json'
        }

        if self.api_key:
            headers['X-API-Key'] = self.api_key

        payload = {
            'model': self.model,
            'messages': messages,
            'stream': False,
            'options': {
                'temperature': self.temperature
            }
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()

            if 'message' in result and 'content' in result['message']:
                return result['message']['content']
            elif 'response' in result:
                return result['response']
            else:
                raise ValueError(f"無法解析 LLM 響應格式: {result}")

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"LLM API 請求失敗: {e}")


class SchoolQASystem:
    """學校規章問答系統"""

    def __init__(self,
                 retriever: Retriever,
                 llm_provider: LLMProvider,
                 reranker: Optional[Reranker] = None):
        self.retriever = retriever
        self.llm_provider = llm_provider
        self.reranker = reranker

        self.SYSTEM_PROMPT = """你是一位嚴謹的法規/規章問答助手。  
請完全依據「參考文獻」提供答案：  
- 若文獻有明確規定，請用中文回答，簡明扼要 (2–4 句為宜)，並在合適時自然提及條號（例如：「依第X條…」）。  
- 若文獻未涵蓋或無明確規定，請直接回答：「文獻未明確說明」。  
- 請勿使用參考文獻以外的知識、推測或自行解釋。"""

    def build_user_prompt(self, question: str, references: List[RetrievalResult]) -> str:
        """構建用戶 prompt"""
        if references:
            ref_texts = []
            for result in references:
                ref_text = f"【{result.document.title}】\n{result.document.content}"
                ref_texts.append(ref_text)
            ref_content = "\n\n".join(ref_texts)
        else:
            ref_content = "（無）"

        return f"""參考文獻：
{ref_content}

問題：
{question.strip()}

請以中文回答，務必精簡清楚，且只根據參考文獻作答。不要加上多餘前後綴。"""

    def answer_question(self, question: str, top_k: int,
                        final_top_k: Optional[int] = None,
                        max_ref_length: int = 8000) -> Dict[str, Any]:
        """回答問題"""
        if final_top_k is None:
            final_top_k = min(top_k, 5)  # 預設最多使用 5 個文檔

        # 步驟 1: 檢索相關文檔
        print(f"正在檢索與問題相關的文檔 (top_k={top_k})...")
        retrieval_results = self.retriever.retrieve(question, top_k=top_k)
        original_count = len(retrieval_results)

        # 步驟 2: 重排序 (如果有 reranker)
        if self.reranker and retrieval_results:
            print("正在對檢索結果進行重排序...")
            retrieval_results = self.reranker.rerank(
                question, retrieval_results)

            if len(retrieval_results) > final_top_k:
                print(f"重排序後取前 {final_top_k} 個文檔")
                retrieval_results = retrieval_results[:final_top_k]
        else:
            if len(retrieval_results) > final_top_k:
                retrieval_results = retrieval_results[:final_top_k]

        # 步驟 3: 截斷參考文獻長度
        total_length = 0
        filtered_results = []
        for result in retrieval_results:
            content_length = len(result.document.title) + \
                len(result.document.content)
            if total_length + content_length <= max_ref_length:
                filtered_results.append(result)
                total_length += content_length
            else:
                remaining_length = max_ref_length - \
                    total_length - len(result.document.title)
                if remaining_length > 100:  # 至少保留100字
                    truncated_content = result.document.content[:
                                                                remaining_length] + "...(截斷)"
                    truncated_result = RetrievalResult(
                        document=Document(
                            id=result.document.id,
                            title=result.document.title,
                            content=truncated_content,
                            metadata=result.document.metadata
                        ),
                        score=result.score,
                        rank=result.rank
                    )
                    filtered_results.append(truncated_result)
                break

        # 步驟 4: 構建 prompt 並調用 LLM
        user_prompt = self.build_user_prompt(question, filtered_results)
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        print("正在生成回答...")
        answer = self.llm_provider.generate(messages)

        # 返回結果
        return {
            "question": question,
            "answer": answer,
            "retrieved_documents": [
                {
                    "id": result.document.id,
                    "title": result.document.title,
                    "score": result.score,
                    "rank": result.rank
                }
                for result in filtered_results
            ],
            "total_retrieved": original_count,
            "used_reranker": self.reranker is not None
        }


def load_documents_from_json(file_path: str) -> List[Document]:
    """從 JSON 檔案載入文檔

    預期格式:
    {
        "L2-26國立中興大學學位學程實施要點（961025）": {
            "1": "第一條 ...",
            "2": "第二條 ...",
            ...
        },
        ...
    }
    """
    documents = []

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for doc_id, doc_data in data.items():
        if isinstance(doc_data, dict):
            # 將條號內容按順序組合成完整文檔
            def sort_key(item):
                key = item[0]
                try:
                    return int(key)
                except ValueError:
                    return float('inf')  # 非數字條號排到最後

            sorted_items = sorted(doc_data.items(), key=sort_key)
            content_parts = [f"{content}" for _, content in sorted_items]
            full_content = "\n".join(content_parts)

            title = doc_id
        else:
            title = doc_id
            full_content = str(doc_data)

        document = Document(
            id=doc_id,
            title=title,
            content=full_content,
            metadata={"original_structure": doc_data if isinstance(
                doc_data, dict) else None}
        )
        documents.append(document)

    return documents


def create_chunked_documents(documents: List[Document],
                             chunk_size: int = 800,
                             chunk_overlap: int = 100) -> List[Document]:
    """將長文檔分割成較小的chunks (可選功能)"""
    chunked_docs = []

    for doc in documents:
        content = doc.content
        if len(content) <= chunk_size:
            chunked_docs.append(doc)
        else:
            start = 0
            chunk_id = 0

            while start < len(content):
                end = min(start + chunk_size, len(content))
                chunk_text = content[start:end]

                chunk_doc = Document(
                    id=f"{doc.id}_chunk_{chunk_id}",
                    title=f"{doc.title} (第{chunk_id+1}段)",
                    content=chunk_text,
                    metadata={
                        "parent_doc_id": doc.id,
                        "chunk_index": chunk_id,
                        "is_chunk": True,
                        **(doc.metadata or {})
                    }
                )
                chunked_docs.append(chunk_doc)

                chunk_id += 1
                if end >= len(content):
                    break

                # 計算下一個chunk的開始位置，考慮overlap
                start = max(end - chunk_overlap, start + 1)

    return chunked_docs


def main():
    try:
        # 1. 初始化 Embedding 提供者（含磁碟快取）
        embedding_provider = CustomEmbeddingProvider(
            api_url=EMBEDDING_API_URL,
            api_key=EMBEDDING_API_KEY
        )

        # 2. 初始化檢索器
        retriever = DenseRetriever(embedding_provider)

        # 3. 載入文檔
        print("正在載入文檔...")
        documents = load_documents_from_json(DOCUMENTS_JSON_PATH)
        print(f"載入了 {len(documents)} 個原始文檔")

        # 4. 可選：對長文檔進行分塊處理
        if USE_CHUNKING:
            print("正在進行文檔分塊處理...")
            documents = create_chunked_documents(
                documents, CHUNK_SIZE, CHUNK_OVERLAP)
            print(f"分塊後共有 {len(documents)} 個文檔片段")

        retriever.add_documents(documents)

        # 5. 初始化 Reranker (如果啟用)
        reranker = None
        if USE_RERANKER:
            try:
                print("正在初始化 Reranker...")
                reranker = BGERerankerV2M3(
                    model_name=RERANKER_MODEL,
                    device=RERANKER_DEVICE,
                    max_length=512,
                    batch_size=16
                )
                print("Reranker 初始化成功")
            except Exception as e:
                print(f"Reranker 初始化失敗: {e}")
                print("將在沒有 reranker 的情況下繼續運行")
                reranker = None

        # 6. 初始化 LLM 提供者
        llm_provider = OllamaProvider(
            api_url=OLLAMA_API_URL,
            model=OLLAMA_MODEL,
            api_key=GPT_OSS_API_KEY,
            temperature=0.2
        )

        # 7. 初始化問答系統
        qa_system = SchoolQASystem(
            retriever=retriever,
            llm_provider=llm_provider,
            reranker=reranker
        )

        print(f"\n{'='*60}")
        print(f"問題：{QUESTION}")
        print('='*60)

        try:
            result = qa_system.answer_question(
                QUESTION,
                top_k=TOP_K,  # 初始檢索數量
                final_top_k=FINAL_TOP_K if reranker else TOP_K  # 最終使用數量
            )

            print(f"回答：{result['answer']}")
            print(f"\n使用的參考文獻 ({len(result['retrieved_documents'])} 個):")
            for i, doc_info in enumerate(result['retrieved_documents'], 1):
                print(
                    f"{i}. {doc_info['title']} (分數: {doc_info['score']:.4f})")

            if reranker and result['total_retrieved'] > len(result['retrieved_documents']):
                print(
                    f"\n註：初始檢索到 {result['total_retrieved']} 個文檔，經 reranker 篩選後使用 {len(result['retrieved_documents'])} 個")

        except Exception as e:
            print(f"處理問題時發生錯誤：{e}")

    except FileNotFoundError:
        print(f"找不到文檔檔案：{DOCUMENTS_JSON_PATH}")
        print("請確保檔案路徑正確，並且檔案格式符合預期")
    except Exception as e:
        print(f"系統初始化失敗：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
