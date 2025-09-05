"""
School Regulation Q&A System with Dense Retriever and Embedding Cache
"""

import json
import math
import os
import pickle
import hashlib
from typing import Dict, List, Tuple, Optional, Any
import requests
from dataclasses import dataclass
from abc import ABC, abstractmethod
from dotenv import load_dotenv

load_dotenv()

# 配置參數 (請根據實際情況修改)
QUESTION = "我現在是碩二，準備要畢業了，我要怎麼跑畢業流程？"
TOP_K = 5  # 檢索返回的文檔數量

EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL")
EMBEDDING_API_KEY = None  # 如果需要的話

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL")
OLLAMA_API_KEY = os.getenv("GPT_OSS_API_KEY")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

DOCUMENTS_JSON_PATH = "./data/school_rules.json"
EMBEDDINGS_CACHE_PATH = "./embedding/embeddings_cache.pkl"  # 新增：緩存文件路徑


# 設定是否使用文檔分塊
USE_CHUNKING = True
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


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


@dataclass
class EmbeddingCacheEntry:
    """Embedding 緩存條目"""
    text_hash: str
    embedding: List[float]
    timestamp: float
    text_preview: str  # 用於調試，存儲文本前100個字符


class EmbeddingCache:
    """Embedding 緩存管理器"""

    def __init__(self, cache_file_path: str):
        self.cache_file_path = cache_file_path
        self.cache: Dict[str, EmbeddingCacheEntry] = {}
        self.load_cache()

    def _get_text_hash(self, text: str) -> str:
        """計算文本的 MD5 哈希值"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def load_cache(self):
        """從文件載入緩存"""
        try:
            if os.path.exists(self.cache_file_path):
                with open(self.cache_file_path, 'rb') as f:
                    self.cache = pickle.load(f)
                print(f"已載入 {len(self.cache)} 個 embedding 緩存條目")
            else:
                print("未找到 embedding 緩存文件，將建立新的緩存")
        except Exception as e:
            print(f"載入 embedding 緩存失敗: {e}")
            self.cache = {}

    def save_cache(self):
        """將緩存保存到文件"""
        try:
            # 確保目錄存在
            os.makedirs(os.path.dirname(self.cache_file_path), exist_ok=True)
            with open(self.cache_file_path, 'wb') as f:
                pickle.dump(self.cache, f)
            print(f"已保存 {len(self.cache)} 個 embedding 緩存條目")
        except Exception as e:
            print(f"保存 embedding 緩存失敗: {e}")

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """從緩存中獲取 embedding"""
        text_hash = self._get_text_hash(text)
        if text_hash in self.cache:
            return self.cache[text_hash].embedding
        return None

    def set_embedding(self, text: str, embedding: List[float]):
        """將 embedding 存入緩存"""
        import time
        text_hash = self._get_text_hash(text)
        self.cache[text_hash] = EmbeddingCacheEntry(
            text_hash=text_hash,
            embedding=embedding,
            timestamp=time.time(),
            text_preview=text[:100] + "..." if len(text) > 100 else text
        )

    def clear_cache(self):
        """清空緩存"""
        self.cache = {}
        if os.path.exists(self.cache_file_path):
            os.remove(self.cache_file_path)
        print("已清空 embedding 緩存")

    def get_cache_stats(self) -> Dict[str, Any]:
        """獲取緩存統計信息"""
        return {
            "total_entries": len(self.cache),
            "cache_file_exists": os.path.exists(self.cache_file_path),
            "cache_file_path": self.cache_file_path
        }


class EmbeddingProvider(ABC):
    """Embedding 提供者抽象基類"""

    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        pass


class CustomEmbeddingProvider(EmbeddingProvider):
    """使用自定義 API 的 Embedding 提供者，支持緩存"""

    def __init__(self, api_url: str, api_key: Optional[str] = None,
                 timeout: int = 30, cache: Optional[EmbeddingCache] = None):
        self.api_url = api_url
        self.api_key = api_key
        self.timeout = timeout
        self.cache = cache
        self.cache_hits = 0
        self.api_calls = 0

    def get_embedding(self, text: str) -> List[float]:
        """獲取文本的向量表示，優先使用緩存"""
        # 先檢查緩存
        if self.cache:
            cached_embedding = self.cache.get_embedding(text)
            if cached_embedding is not None:
                self.cache_hits += 1
                return cached_embedding

        # 緩存中沒有，調用 API
        self.api_calls += 1
        embedding = self._call_api(text)

        # 將結果存入緩存
        if self.cache and embedding:
            self.cache.set_embedding(text, embedding)

        return embedding

    def _call_api(self, text: str) -> List[float]:
        """調用 API 獲取 embedding"""
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }

        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'

        data = {'inputs': text}

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()

            # 處理不同的響應格式
            if isinstance(result, list) and len(result) > 0:
                return result[0] if isinstance(result[0], list) else result
            elif "embeddings" in result:
                embeddings = result["embeddings"]
                return embeddings[0] if isinstance(embeddings[0], list) else embeddings
            elif "embedding" in result:
                return result["embedding"]
            else:
                # 嘗試其他可能的字段
                for key in ["vectors", "data", "result"]:
                    if key in result:
                        vec = result[key]
                        return vec[0] if isinstance(vec, list) and isinstance(vec[0], list) else vec

                raise ValueError(f"無法解析 embedding 響應格式: {result}")

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Embedding API 請求失敗: {e}")

    def get_stats(self) -> Dict[str, int]:
        """獲取統計信息"""
        return {
            "cache_hits": self.cache_hits,
            "api_calls": self.api_calls,
            "total_requests": self.cache_hits + self.api_calls
        }


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
                # 可以選擇只對標題建索引，或標題+內容
                # 限制長度避免超限
                text_to_embed = f"{doc.title}\n{doc.content[:1000]}"
                embedding = self.embedding_provider.get_embedding(
                    text_to_embed)
                self.document_embeddings.append(embedding)

                if (i + 1) % 10 == 0:
                    print(f"已處理 {i + 1}/{len(self.documents)} 個文檔")

            except Exception as e:
                print(f"文檔 {doc.id} 建立索引失敗: {e}")
                # 添加零向量作為占位符
                self.document_embeddings.append([0.0] * 768)  # 假設向量維度為768

        self._indexed = True
        print("向量索引構建完成")

        # 如果使用了緩存，保存緩存
        if hasattr(self.embedding_provider, 'cache') and self.embedding_provider.cache:
            self.embedding_provider.cache.save_cache()

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

        # 計算相似度
        similarities = []
        for i, doc_embedding in enumerate(self.document_embeddings):
            similarity = self.cosine_similarity(query_embedding, doc_embedding)
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
    """重排序器抽象基類 (預留擴展)"""

    @abstractmethod
    def rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        pass


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

            # 解析響應
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
                        max_ref_length: int = 8000) -> Dict[str, Any]:
        """回答問題"""
        # 步驟 1: 檢索相關文檔
        print(f"正在檢索與問題相關的文檔 (top_k={top_k})...")
        retrieval_results = self.retriever.retrieve(question, top_k=top_k)

        # 步驟 2: 重排序 (如果有 reranker)
        if self.reranker and retrieval_results:
            print("正在對檢索結果進行重排序...")
            retrieval_results = self.reranker.rerank(
                question, retrieval_results)

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
            "total_retrieved": len(retrieval_results)
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
                """排序條號，處理數字排序"""
                key = item[0]
                try:
                    return int(key)
                except ValueError:
                    return float('inf')  # 非數字條號排到最後

            sorted_items = sorted(doc_data.items(), key=sort_key)
            content_parts = [f"{content}" for _, content in sorted_items]
            full_content = "\n".join(content_parts)

            # 使用文檔ID作為標題，如果需要可以進一步處理提取更好的標題
            title = doc_id

        else:
            # 如果不是字典格式，直接當作內容處理
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
            # 文檔不長，直接保留
            chunked_docs.append(doc)
        else:
            # 分割長文檔
            chunks = []
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
        # 1. 初始化 Embedding 緩存
        embedding_cache = EmbeddingCache(EMBEDDINGS_CACHE_PATH)
        print(f"緩存統計: {embedding_cache.get_cache_stats()}")

        # 2. 初始化 Embedding 提供者（帶緩存）
        embedding_provider = CustomEmbeddingProvider(
            api_url=EMBEDDING_API_URL,
            api_key=EMBEDDING_API_KEY,
            cache=embedding_cache
        )

        # 3. 初始化檢索器
        retriever = DenseRetriever(embedding_provider)

        # 4. 載入文檔
        print("正在載入文檔...")
        documents = load_documents_from_json(DOCUMENTS_JSON_PATH)
        print(f"載入了 {len(documents)} 個原始文檔")

        # 5. 可選：對長文檔進行分塊處理
        if USE_CHUNKING:
            print("正在進行文檔分塊處理...")
            documents = create_chunked_documents(
                documents, CHUNK_SIZE, CHUNK_OVERLAP)
            print(f"分塊後共有 {len(documents)} 個文檔片段")

        retriever.add_documents(documents)

        # 6. 初始化 LLM 提供者
        llm_provider = OllamaProvider(
            api_url=OLLAMA_API_URL,
            model=OLLAMA_MODEL,
            api_key=OLLAMA_API_KEY,
            temperature=0.2
        )

        # 7. 初始化問答系統
        qa_system = SchoolQASystem(
            retriever=retriever,
            llm_provider=llm_provider
            # reranker=None  # 可以在這裡添加 reranker
        )

        print(f"\n{'='*60}")
        print(f"問題：{QUESTION}")
        print('='*60)

        try:
            result = qa_system.answer_question(QUESTION, top_k=TOP_K)
            print(f"回答：{result['answer']}")
            print(f"\n檢索到的參考文獻 ({result['total_retrieved']} 個):")
            for i, doc_info in enumerate(result['retrieved_documents'], 1):
                print(
                    f"{i}. {doc_info['title']} (相似度: {doc_info['score']:.4f})")

            # 顯示 embedding 統計信息
            print(f"\nEmbedding 統計:")
            stats = embedding_provider.get_stats()
            print(f"  緩存命中: {stats['cache_hits']}")
            print(f"  API 調用: {stats['api_calls']}")
            print(f"  總請求: {stats['total_requests']}")
            if stats['total_requests'] > 0:
                cache_hit_rate = stats['cache_hits'] / \
                    stats['total_requests'] * 100
                print(f"  緩存命中率: {cache_hit_rate:.1f}%")

        except Exception as e:
            print(f"處理問題時發生錯誤：{e}")

    except FileNotFoundError:
        print(f"找不到文檔檔案：{DOCUMENTS_JSON_PATH}")
        print("請確保檔案路徑正確，並且檔案格式符合預期")
    except Exception as e:
        print(f"系統初始化失敗：{e}")


if __name__ == "__main__":
    main()
