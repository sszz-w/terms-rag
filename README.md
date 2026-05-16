# 条款要素检索 RAG 系统

## 方法论

本系统基于 **语义向量检索（Retrieval-Augmented Generation）** 架构，用于从招标条款中自动匹配对应的证书/报告要求。

### 核心思路

1. **文本向量化**：将每条原始条款通过中文 Embedding 模型（BAAI/bge-small-zh-v1.5）转换为 512 维语义向量
2. **向量存储**：将向量及其关联的元数据（证书名称、内容要素、有效性判定）存入 ChromaDB 向量数据库
3. **语义检索**：用户输入条款文本 → 生成查询向量 → 余弦相似度匹配 → 返回最相似条款的关联信息

### 技术选型

| 组件 | 选择 | 理由 |
|------|------|------|
| Embedding 模型 | BAAI/bge-small-zh-v1.5 | 中文语义理解能力强，模型体积小（~100MB） |
| 推理引擎 | ONNX Runtime (via fastembed) | 无需 GPU，CPU 即可高效运行 |
| 向量数据库 | ChromaDB | 纯 Python，持久化存储，支持百万级文档扩展 |
| 相似度度量 | 余弦相似度 | 适合文本语义匹配场景 |

### 工作流程

```
用户输入条款 → Embedding 模型编码 → 向量相似度检索 → 返回 Top-K 匹配结果
                                                         ├── 证书/报告名称
                                                         ├── 内容要素字段
                                                         └── 有效性判定
```

### 特点

- 支持模糊匹配：条款中金额、年限、数量等细节变动不影响匹配准确性
- 无需 LLM 生成：直接返回结构化字段，结果确定性高
- 离线运行：模型和数据均在本地，无需联网或 API Key

---

## 环境要求

- macOS / Linux / Windows
- Python 3.12（推荐）
- 磁盘空间：约 500MB（模型缓存 + 依赖）
- 内存：2GB 以上

---

## 安装部署

### 1. 创建虚拟环境

```bash
# macOS Homebrew 安装的 Python 3.12
/usr/local/opt/python@3.12/bin/python3.12 -m venv .venv

# 或通用方式
python3.12 -m venv .venv
```

### 2. 激活环境

```bash
source .venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 导入数据

将数据文件 `条款要素抽取.xlsx` 放在项目根目录，然后执行：

```bash
python ingest.py
```

首次运行会自动下载 Embedding 模型（约 100MB），后续使用本地缓存。

---

## 使用方式

### 交互模式

```bash
source .venv/bin/activate
python query.py
```

输入条款内容后回车，输入 `q` 退出。

### 单次查询

```bash
python query.py "投标方需要具备ISO9001认证"
```

### 输出示例

```
============================================================
匹配 #1  (相似度: 0.8312)
============================================================
原始条款: 体系认证要求：投标单位需通过ISO9001质量体系认证。

证书/报告名称: ISO9001质量管理体系认证证书

内容要素字段: 证书编号、企业名称、认证范围、认证标准号、证书有效期、认证机构名称

有效性判定:
企业名称与投标人名称一致；
认证范围包含电气设备；
认证标准号包含ISO9001；
证书有效期在投标截止日期后；
```

---

## 数据更新

当 Excel 数据有变更时，重新运行导入即可（会自动覆盖旧数据）：

```bash
source .venv/bin/activate
python ingest.py
```

---

## 配置调整

编辑 `config.py`：

```python
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"  # Embedding 模型
COLLECTION_NAME = "clauses"                   # 向量库集合名
TOP_K = 3                                     # 返回匹配数量
```

---

## 项目结构

```
rag/
├── config.py          # 配置参数
├── embedding.py       # Embedding 模块
├── ingest.py          # 数据导入
├── query.py           # 查询接口
├── requirements.txt   # Python 依赖
├── chroma_db/         # 向量数据库（自动生成）
├── model_cache/       # 模型缓存（自动生成）
└── 条款要素抽取.xlsx    # 源数据文件
```

---

## 扩展方向

- **Web 接口**：加 FastAPI/Gradio 提供 HTTP 查询服务
- **Reranker**：引入交叉编码器对 Top-K 结果二次排序，提升精度
- **LLM 生成**：接入大模型对检索结果做自然语言总结
- **批量查询**：支持从文件批量读取条款并输出匹配结果
