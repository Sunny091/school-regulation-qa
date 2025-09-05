# 📘 School Regulation Q&A — Dense Retriever (with/without Reranker)

以 Dense Retriever 為核心的學校規章問答系統。支援兩種執行模式：

-   **DenseOnly**：單純用 embedding 做語意檢索
-   **DenseWithReranker**：先 Dense 檢索，再用 BGE Reranker 重新排序提升準確度

系統會將規章 JSON 檔載入、（可選）分塊、向量化、相似度檢索，並把 Top-K 文本做為參考文獻交給 LLM 產生中文回答。Embedding 結果會快取到 `./embedding/embeddings_cache.pkl` 以加速重複執行。

## 🧭 兩種模式的差異

| 項目           | DenseOnly                                                | DenseWithReranker                                                 |
| -------------- | -------------------------------------------------------- | ----------------------------------------------------------------- |
| 檔案/入口      | `./DenseOnly/XXX.py`（XXX='gptoss or claude or gemini'） | `./DenseWithReranker/main.py`（XXX='gptoss or claude or gemini'） |
| 規章 JSON 路徑 | `./data/school_rules.json`                               | `./data/school_rules.json`                                        |
| 依賴           | `requests`, `python-dotenv`                              | 另需 `transformers`, `torch`（或 `FlagEmbedding`）                |
| 效果           | 輕量、啟動快                                             | 檢索更準確（Top-K 重新排序）                                      |
| 適用情境       | 文件量小或快速試跑                                       | 文件中常有語義近似段落、需要更準確排名                            |

你也可以只保留一個入口檔，用 **環境變數或常數** 切換 `USE_RERANKER=True/False` 與 `DOCUMENTS_JSON_PATH`。

## 📂 專案結構建議

```
.
├── data/
│   └── school_rules.json            # 規章 JSON
├── embedding/
│   └── embeddings_cache.pkl         # 向量快取（自動建立/更新）
├── DenseOnly/
│   ├── gptoss.py                    # 使用 gpt-oss 當 generator（不含 reranker）
│   ├── claude.py                    # 使用 claude 當 generator （不含 reranker）
│   └── gemini.py                    # 使用 gemini 當 generator （不含 reranker）
├── DenseWithReranker/
│   ├── gptoss.py                    # 使用 gpt-oss 當 generator（含 reranker）
│   ├── claude.py                    # 使用 claude 當 generator （含 reranker）
│   └── gemini.py                    # 使用 gemini 當 generator （含 reranker）
├── .env                             # 參數設定
└── README.md
```

## 🔧 安裝與環境

### 1) 安裝依賴

**DenseOnly 最小依賴**：

```bash
pip install requests python-dotenv
```

**DenseWithReranker 另外需要其一**：

**A. 使用 Transformers 版本**：

```bash
pip install transformers torch
```

**B. 或使用 FlagEmbedding（速度佳）**：

```bash
pip install FlagEmbedding
```

兩者擇一；程式會自動偏好 FlagEmbedding，若未安裝會 fallback 到 transformers + torch。

### 2) 設定 .env

在專案根目錄建立 `.env`：

```env
ANTHROPIC_API_KEY
GOOGLE_API_KEY
GPT_OSS_API_KEY
EMBEDDING_API_URL
OLLAMA_API_URL
OLLAMA_MODEL
RERANKER_MODEL
```

## 🗂️ 規章 JSON 格式

```json
{
    "L2-26國立中興大學學位學程實施要點（961025）": {
        "1": "第一條  ……",
        "2": "第二條  ……",
        "3": "第三條  ……"
    },
    "L2-27國立中興大學其他規章": {
        "1": "第一條 ……",
        "2": "第二條 ……"
    }
}
```

以「文件標題」為 key；value 是「條文號 → 條文內容」的物件。

系統會把條文依數字順序串成一個完整文件；也支援啟用 **分塊（chunking）** 以提升檢索粒度。

## 🚀 執行方式

### DenseOnly

預設文件路徑：`./data/school_rules.json`

```bash
python ./DenseOnly/XXX.py
```

### DenseWithReranker

預設文件路徑：`./data/school_rules.json`

```bash
python DenseWithReranker/XXX.py
```

執行時會：

1. 載入 JSON 並（可選）分塊
2. 對每個（片段）文本建立/讀取 embedding（自動快取）
3. 檢索 Top-K；如啟用 reranker 再重排
4. 將最終參考文獻送入 LLM 產生中文回答

## ⚙️ 重要參數

在程式內部：

```python
# 檢索
TOP_K = 50              # 初次檢索數量
FINAL_TOP_K = 5         #（有 reranker 時）最終給 LLM 的文檔數量
MAX_REF_LENGTH = 8000   # 參考文獻字數上限（避免 prompt 爆長）

# Reranker
USE_RERANKER = True/False
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_DEVICE = "cuda"  # 可改為 'cpu' / 'mps'

# 分塊
USE_CHUNKING = True
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# 快取
EMBEDDING_CACHE_PATH = "./embedding/embeddings_cache.pkl"
```

### 小技巧：

-   文件很長時建議 `USE_CHUNKING=True`
-   文件少、實驗快：`TOP_K=30`、`FINAL_TOP_K=5` 即可
-   Reranker 在多個相似段落時提升明顯

## 🧠 回答風格（System Prompt 準則）

-   嚴格根據「參考文獻」作答
-   有條號時自然提及（例：「依第 X 條…」）
-   文獻未涵蓋就回覆「文獻未明確說明」
-   2–4 句、中文精簡回答；不加多餘前後綴

## 💾 向量快取（超重要）

-   **路徑**：`./embedding/embeddings_cache.pkl`
-   **Key**：文字內容 SHA256（換行正規化 + trim）
-   **嵌入成功會即時寫回**；程式中斷不會遺失
-   **好處**：大量減少重算時間，再次啟動更快

**重置快取**：刪除 `./embedding/embeddings_cache.pkl` 即可

（或你也可用版本前綴 `v1:` 作為 key namespace，之後升級為 `v2:` 強制重新計算）

## 🧪 測試問題（內建示例）

-   「我現在是碩二，準備要畢業了，我要怎麼跑畢業流程？」
-   「學位學程的招生名額有什麼限制？」
-   「如何申請人文學及社會科學學術著作出版？」

你可以改成 CLI 參數或 HTTP 介面（未內建）以便對接前端。

## 🛠️ 常見問題（FAQ / Troubleshooting）

### Q1：執行就報 未設定 EMBEDDING_API_URL 或 LLM URL/Model？

**A**：確認 `.env` 是否存在，且鍵名拼對；或直接在程式中硬寫測試值。

### Q2：Reranker 抱怨 transformers/torch 未安裝？

**A**：若不需要 reranker，請使用 DenseOnly；若需要，請 `pip install transformers torch` 或改裝 FlagEmbedding。

### Q3：相似度都很低或結果不穩定？

**A**：確認分塊是否合理（CHUNK_SIZE/OVERLAP），以及 TOP_K 是否過小。嘗試啟用 Reranker。

### Q4：維度不一致或 embedding 解析錯誤？

**A**：你的 Embedding API 回傳格式與維度需一致；若切換模型建議清空快取後重建。

### Q5：快取文件損壞（pickle error）？

**A**：刪除 `./embedding/embeddings_cache.pkl` 後重跑；或改變 key namespace（例如 `v2:`）。

## 📈 建議評估方式

-   **Top-K 命中率**：檢查最終引用文獻是否涵蓋正確條文
-   **人工標註準確率**：抽樣問題比對答案與原文是否一致
-   **Reranker 影響**：同一問題對照 DenseOnly 與 WithReranker 的 Top-1/Top-3 一致性

## ✅ 快速檢查清單

-   [ ] `.env` 設好 Embedding 與 LLM 端點
-   [ ] `school_rules.json` 放在指定路徑
-   [ ] 首次跑完看到 `embedding/embeddings_cache.pkl`
-   [ ] DenseOnly 能跑通
-   [ ] DenseWithReranker 能跑通（必要套件已裝）
-   [ ] 答案能自然引用條號，文獻不足時會明確說「未明確說明」
