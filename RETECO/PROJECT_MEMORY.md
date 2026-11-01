# RETECO 项目记忆

更新时间：2026-10-20

## 服务器实测项目状态（2026-10-20）

已通过 SSH 登录服务器 `k8s.sgurad.com:12025`，用户为 `huangliang`，确认远程项目根目录为：

```text
/home/huangliang/reteco/
├── RETECO/                         代码、实验脚本、checkpoint、run 和诊断结果
└── reteco_data/                    Track 1 数据
    └── track1_tempo/
        ├── bitcoin/
        ├── cardano/
        ├── economics/
        ├── genealogy/
        ├── history/
        ├── hsm/
        ├── iota/
        ├── law/
        ├── monero/
        ├── politics/
        ├── quant/
        ├── travel/
        └── workplace/
```

服务器上的 `/home/huangliang/reteco/RETECO` 目前包含：

```text
RETECO/
├── cache/                           文档向量缓存
├── checkpoints/                     LoRA 和 Sentence Transformers checkpoint
├── diagnostics/                    已完成 run 的评测和召回诊断
├── docs/                            项目网站文档
├── exp/                             实验脚本、run 文件和实验日志
├── finetune_data/                   微调 JSON 数据
├── logs/                            训练、检索和重排日志
├── runs/                            TREC 六列格式的检索结果
├── starter_kit/                     官方格式检查和评分器
├── PROJECT_MEMORY.md
└── README.md / main.py / proposal PDF
```

基础模型已在服务器本地准备好：

- `/home/huangliang/models/Qwen3-Reranker-8B`，约 16 GB。
- `/home/huangliang/models/Qwen3-Embedding-8B`，约 15 GB。
- 服务器为 2 张 NVIDIA RTX 3090，每张 24 GB；实测记录时 GPU 空闲。

## 已完成事项（服务器实测）

- BM25 Top-500：`runs/runs_bm25_top500/`，13 个领域各有 train/dev，共 26 个 run 文件。
- BGE-large Top-500：`runs/runs_bge_large_top500/`，13 个领域各有 train/dev，共 26 个 run 文件。
- Qwen3-Embedding-8B Top-500：`runs/runs_qwen3_embedding_8b_top500/`，共 26 个 run 文件；对应评测在 `diagnostics/qwen3_embedding_8b_top500/`。
- Qwen3-ST LoRA v4 Top-500：`runs/runs_qwen3_st_bf16_lora_v4_top500/`，共 26 个 run 文件；对应评测在 `diagnostics/qwen3_st_bf16_lora_v4_top500/`。
- Qwen3-ST LoRA v1、v2、v3、v4 的最终 checkpoint 位于 `checkpoints/qwen3_st_bf16_lora_v{1,2,3,4}/final/`；这些目录包含 `modules.json`、`1_Pooling/`、`2_Normalize/` 和 PEFT adapter 文件。
- Qwen3 Reranker LoRA v1 已完成完整训练：`checkpoints/qwen3_reranker_lora_v1/`。训练使用 1211 个查询、5 个 epoch、每组 4 个 hard negative、`max_length=4096`、LoRA `r=16`、`alpha=32`、`dropout=0.1`、双 GPU、共 380 次 optimizer update。
- Reranker v1 已保存 `epoch_001`、`epoch_002`、`epoch_003`、`epoch_004`、`epoch_005` 和 `final`；每个 checkpoint 含 `adapter_config.json` 和 `adapter_model.safetensors`。epoch 1 loss 为约 `0.29095`，epoch 5 loss 为约 `0.12649`。
- Reranker v1 训练结果记录在 `checkpoints/qwen3_reranker_lora_v1/training_config.json`、`training_log.jsonl` 和 `training_result.json`；训练共 380 次更新，约耗时 13474.7 秒。
- 已新增 `exp/run_qwen3_reranker_peft_1a.py`：使用基础 Qwen3-Reranker 加载 `PeftModel.from_pretrained()`，按 yes/no 条件概率对已有 TREC run 的候选进行 Cross-Encoder 重排，支持 `final`、任意 `epoch_00N`、`--resume` 和按 GPU 分领域运行。

## 尚未完成事项（截至 2026-10-20）

- Qwen3 Reranker LoRA v1 尚未在 Track 1a 的 `dev` 或 `train` 上生成 Cross-Encoder rerank run；`runs/` 中目前没有 `qwen3_reranker_lora_*` 输出目录。
- `logs/qwen_reranker8b/` 中已有的是旧的基础 Qwen3-Reranker `iota` 实验日志和分数缓存，不是 LoRA v1 的完整实验结果。
- `exp/run_qwen3_reranker_lora_1a.py` 实际是 Sentence Transformers 向量检索脚本，要求 `modules.json`，不能用于 `train_qwen3_reranker_1a.py` 产生的 PEFT Cross-Encoder checkpoint；LoRA Cross-Encoder 应使用 `exp/run_qwen3_reranker_peft_1a.py`。
- LoRA Reranker 的下一步实验建议先用 `qwen3_st_bf16_lora_v4` 的 Bitcoin dev Top-500 作为输入，分别测试 `epoch_001` 到 `epoch_005` 和 `final`，先用 `--rerank-depth 100` 做 smoke test，再决定是否扩大到 300/500 或运行全部 13 个领域。


## 项目位置

### 本机项目

- Windows 工作目录：`H:\code_python\reteco`
- 官方仓库：`H:\code_python\reteco\RETECO`
- Starter Kit：`H:\code_python\reteco\RETECO\starter_kit`
- 实验代码：`H:\code_python\reteco\RETECO\exp`
- 网站文档：`H:\code_python\reteco\RETECO\docs`
- BM25 结果：`H:\code_python\reteco\RETECO\starter_kit\runs_release`
- BGE 结果：`H:\code_python\reteco\RETECO\exp\runs_dense`
- 数据目录：`H:\code_python\reteco\reteco_data`；2026-10-06 本机仅看到 `history` 和 `iota`，服务器 1a 的 13 个领域 train/dev 数据已齐全。

### 远程 Linux 环境

- Conda 环境名：`reteco`
- Conda 环境路径：`/home/huangliang/.conda/envs/reteco`
- 远程 shell 工作路径示例：`~/reteco/RETECO/starter_kit`
- 远程项目根目录：`/home/huangliang/reteco/RETECO`
- 远程数据根目录：`/home/huangliang/reteco/reteco_data`
- 本机和服务器采用相同的相对目录结构：工作目录下并列放置 `RETECO` 仓库和 `reteco_data` 数据。
- `Z:\Codex\读论文` 是论文阅读和辅助脚本目录，不是 RETECO 项目根目录；项目代码、数据和项目记忆以 `H:\code_python\reteco` 为准。
- 本机与服务器的相对目录结构（服务器将根目录替换为 `/home/huangliang/reteco`）：

  ```text
  reteco/
  ├── RETECO/                       官方代码仓库
  │   ├── .github/workflows/
  │   ├── .idea/
  │   ├── docs/                     网站页面和 sample_data
  │   ├── exp/
  │   │   ├── runs_dense/
  │   │   ├── dense_retrieval.py
  │   │   └── make_t1_table.py
  │   ├── starter_kit/
  │   │   ├── runs_release/
  │   │   ├── bm25.py, bm25_baseline.py, run_release.py
  │   │   ├── build_release.py, download_raw.py
  │   │   ├── official_baseline.py, scorer.py
  │   │   ├── ir_metrics.py, format_checker.py
  │   │   └── requirements.txt, README.md, DATASET_CARD.md,
  │   │       BASELINE_RESULTS.md
  │   ├── main.py
  │   ├── PROJECT_MEMORY.md
  │   └── RETECO_SemEval_2027_Proposal.pdf
  └── reteco_data/                   本机当前仅见 history、iota；服务器 1a 数据齐全
      └── track1_tempo/
          └── iota/
  ```

- SSH 主机：`k8s.sgurad.com`
- SSH 端口：`12025`
- SSH 用户：`huangliang`
- SSH 私钥路径（本机）：`C:\Users\86177\key2`
- 项目远程路径：`~/reteco/RETECO`（用户已确认）。

不要把私钥内容写入仓库；这里只记录连接所需的路径和主机信息。

## 当前目标

当前准备参加 RETECO **Sub-track 1a：Temporal Retrieval**。

- 输入：一个时间约束查询和对应领域的 `documents.jsonl`。
- 输出：按相关性排序的文档 ID，使用 TREC 六列格式。
- 官方排名指标：`nDCG@10`。
- 诊断指标：`Temporal Precision@10`、`Temporal Relevance@10`、`Temporal Coverage@10`、`nDCG|FC@10`。

## 环境摘要

Python：`3.10.21`

### 已记录的关键包

- `torch 2.11.0+cu128`
- `torchvision 0.26.0+cu128`
- `torchaudio 2.11.0+cu128`
- `transformers 5.17.0`
- `sentence-transformers 6.1.0`
- `flagembedding 1.4.2`
- `accelerate 1.15.0`
- `datasets 5.0.1`
- `huggingface-hub 1.32.0`
- `faiss-cpu 1.15.1`
- `numpy 2.2.6`
- `scipy 1.15.3`
- `scikit-learn 1.7.2`
- `pandas 2.3.3`
- `pyarrow 25.0.1`
- `gensim 4.4.0`
- `pytrec-eval-terrier 0.5.10`
- `rank-bm25 0.2.2`
- `tqdm 4.70.1`
- `openai 3.17.0`

### GPU / CUDA

- 服务器内存：`64 GB`（用户于 2026-10-04 提供）。
- GPU：`2 x NVIDIA RTX 3090`，每卡 24 GB 显存（用户于 2026-10-04 提供）。
- CUDA Toolkit：`12.8.1`
- NVIDIA CUDA 12 运行库已安装，包括 cuBLAS、cuDNN、cuSPARSE 等。
- `exp/dense_retrieval.py` 的默认批次为 64；IOTA dev 已以批次 64 成功运行。跑全部 1a 时可先保持 64，再在较大领域试 128，并用 `nvidia-smi` 观察显存和吞吐；若 OOM 则降到 64 或 32。批次大小主要影响速度和显存，不是提高 nDCG 的手段。
- 当前脚本只向 FlagEmbedding 传入 `device="cuda"`，没有显式把不同领域分配到两张 GPU。使用双卡时可让两个独立进程分别设置 `CUDA_VISIBLE_DEVICES=0` 和 `CUDA_VISIBLE_DEVICES=1`，避免同时写入同一领域的缓存或结果文件。

### 运行注意

- `exp/dense_retrieval.py` 依赖 PyTorch、FlagEmbedding，可使用 CUDA。
- `starter_kit/bm25_baseline.py` 是无额外检索依赖的纯 Python 基线。
- `starter_kit/official_baseline.py` 需要 Pyserini 和 JDK；在已粘贴的环境列表中没有确认 `pyserini` 和 JDK，因此运行官方 Lucene BM25 前要单独检查：

  ```bash
  python -c "import pyserini; print('pyserini ok')"
  java -version
  ```

- `pytrec-eval-terrier` 已记录在环境中；如代码要求导入 `pytrec_eval`，需要实际测试导入名是否可用。

## 项目入口

- 纯 Python BM25：`starter_kit/bm25_baseline.py`
- BM25 实现：`starter_kit/bm25.py`
- 官方 BM25：`starter_kit/official_baseline.py`
- 批量运行与评分：`starter_kit/run_release.py`
- 评分器：`starter_kit/scorer.py`
- 提交格式检查：`starter_kit/format_checker.py`
- 稠密检索实验：`exp/dense_retrieval.py`
- 结果表生成：`exp/make_t1_table.py`

## 1a 推荐工作流

### 当前进度（2026-10-06）

- 已完成 1a 全部 13 个领域的 BGE 稠密检索：train 和 dev 各 13 个 run 文件，共 26 个，保存在 `RETECO/exp/runs_dense/`。模型为 `BAAI/bge-base-en-v1.5`，使用 CUDA，Top-100。
- 已完成 1a 全部 13 个领域的 BM25 检索：train 和 dev 各 13 个 run 文件，共 26 个，保存在 `RETECO/exp/runs_bm25/`。
- 已完成 1a 全部 13 个领域的 E5-Mistral 检索：train 和 dev 各 13 个 run 文件，共 26 个，保存在 `RETECO/exp/runs_e5_mistral/`。双卡任务使用 `intfloat/e5-mistral-7b-instruct`、`batch=1`、`max-length=512`，运行约 14 小时 7 分钟，于 2026-10-06 08:46（新加坡时间）完成。
- 三套结果均已通过格式检查和官方评分器，诊断结果分别保存在：`RETECO/exp/diagnostics_bm25/`、`RETECO/exp/diagnostics_dense/`、`RETECO/exp/diagnostics_e5_mistral/`。每套均包含 CSV、JSON 汇总、逐领域格式报告和逐领域评分 JSON。
- 1a dev 宏平均结果：BM25 nDCG@10=`0.1057`、TC@10=`0.1874`；BGE nDCG@10=`0.2067`、TC@10=`0.3901`；E5-Mistral nDCG@10=`0.0876`、TC@10=`0.1061`。
- 1a train 宏平均结果：BM25 nDCG@10=`0.0874`、TC@10=`0.1664`；BGE nDCG@10=`0.1704`、TC@10=`0.3433`；E5-Mistral nDCG@10=`0.1037`、TC@10=`0.1329`。
- 当前结论：BGE 是最好的单一基线；BM25 在部分领域仍有互补价值；当前 E5-Mistral 直接排序低于 BGE 和 BM25，但在 `monero`、`quant`、`travel` dev 上分别达到 `0.1349`、`0.3311`、`0.2413`，可作为补充召回器或待审计模型。
- 候选池分析（dev Top-100 正例召回）：BM25=`0.199`，BGE=`0.539`，BM25+BGE 并集=`0.570`，E5=`0.095`，三者并集=`0.609`。这表明 BGE 是主召回器，BM25 和 E5 可以增加候选覆盖，但 E5 不适合作为当前主排序器。
- E5 结果需要单独做实现审计：论文 TEMPO 报告的 E5 平均 nDCG@10 约为 `0.304`，明显高于当前结果。优先核对官方 pooling、EOS / last-token 处理、instruction 格式、模型版本、`max-length=512` 截断影响，以及文档/query 编码方式。暂不把当前 E5 低分解释为模型本身能力不足。
- 目前未完成：BM25+BGE/E5 融合、时间字段抽取、时间感知重排、覆盖感知 Top-10 选择、监督式 reranker 和领域路由调优。

```text
documents.jsonl + examples_{train,dev}.jsonl
        -> BM25 / BGE / E5 多路召回
        -> 候选集合合并
        -> 主题、实体、事件、时间特征
        -> reranker / 覆盖感知重排
        -> TREC run 文件
        -> scorer.py + qrels_{train,dev}.txt
        -> nDCG@10 和时间诊断指标
```

### 1a 推荐实验路线（已重新设计）

论文和当前结果共同表明，1a 的核心难点不是单纯的语义相似度，而是同时满足主题、实体、事件、时间对齐和多时期证据覆盖。实验应优先解决候选召回和时间对齐，再做复杂训练。

| 阶段 | 方法 | 目的 | 当前优先级 |
| --- | --- | --- | --- |
| 0 | 可复现性与 E5 实现审计 | 解释 E5 与论文结果的异常差距，避免错误实现影响后续结论 | 最高 |
| 1 | BM25、BGE、E5 Top-100 候选分析 | 判断正例是否进入候选池，区分召回问题和排序问题 | 最高 |
| 2 | BM25+BGE 融合；可选加入 E5 | 利用词法和语义互补，提高候选召回 | 最高 |
| 3 | 时间、实体、事件、时期字段抽取 | 为每个 query/document 建立可解释的时间表示 | 高 |
| 4 | 时间感知重排序与覆盖感知 Top-10 | 提升时间相关性、TC 和最终 nDCG@10 | 高 |
| 5 | 多查询/结构化查询改写 | 分别检索原始、基准时期、比较时期、变化/演化证据 | 中高 |
| 6 | Cross-Encoder 或 LLM reranker | 对 Top-100/200 候选联合判断主题、实体、事件和时间 | 中高 |
| 7 | 监督式 reranker | 使用普通负例、时间 hard negative、覆盖 hard negative 学习排序 | 中 |
| 8 | 领域路由和小规模配置调优 | 只在最后利用领域差异提高宏平均，避免过早过拟合 | 最后 |

#### 各阶段实施原则

1. **先审计 E5**：在 `history`、`bitcoin`、`quant`、`monero`、`travel` 上比较 `max-length=512/1024`、官方 pooling 和 instruction；先做 dev 小规模实验，不立即全量重跑。
2. **先做融合再做训练**：使用 BM25、BGE、E5 的 Top-100 结果做 RRF 或归一化加权。初始权重可设 BGE `0.60`、BM25 `0.25`、E5 `0.15`，再在 train 上调参，dev 只做留出验证。
3. **把时间建模显式化**：抽取 query 的 temporal intent、time anchors、events、reasoning class，以及 document 的 time scope、temporal signals、events；不要只依赖 embedding 相似度。
4. **重排必须同时看单篇相关性和集合覆盖**：时间分数要判断时期是否匹配、事件/实体是否匹配、是否只是无关日期；选择 Top-10 时优先补充尚未覆盖的必要时期，但不能为了 TC 引入主题不相关文档。
5. **查询改写使用多个短查询**：原始查询、实体+事件、基准时期、比较时期、变化/演化查询分别召回后合并。不要把长的 LLM 推理链直接拼给 BM25，避免关键词稀释。
6. **监督训练使用正确 hard negatives**：时间错误但主题相关、只覆盖一个时期、时间模糊或过时的文档应作为主要难负例；按 query 划分训练/验证，避免文档或问题泄漏。
7. **所有实验统一记录**：nDCG@10 为主指标，同时报告 TP@10、TR@10、TC@10、NDCG|FC@10、完整覆盖数量、运行时间和显存；报告 train/dev 宏平均及逐领域结果。

#### 推荐的最终系统结构

```text
query
  -> 时间/实体/事件分析
  -> BM25 + BGE + E5 多路召回
  -> 候选集合合并（Top-100/200）
  -> 主题、实体、事件、时间特征计算
  -> Cross-Encoder / 监督 reranker
  -> 覆盖感知的 Top-10 选择
```

当前最值得投入的主线是 **BM25+BGE+时间感知重排**。E5 先作为补充召回器，只有通过实现审计或长度/指令实验证明有效后，才提升其权重。

开发顺序：

1. 保留 BM25、BGE、E5 三套已评分结果作为固定基线；
2. 完成 E5 小规模实现审计；
3. 实现 BM25+BGE（再加 E5）的候选合并与 RRF；
4. 在候选上增加时间、实体、事件和时期覆盖特征；
5. 先做无监督时间感知重排，再做 Cross-Encoder/监督 reranker；
6. 最后进行领域配置调优，并用未参与调参的 dev 结果做最终判断。


## 环境来源

以上环境信息来自用户提供的 `conda list` 文本，原始附件位于：

`C:\Users\86177\.codex\attachments\5bac31a5-36b5-4fd3-8b2a-782b087d59f5\已粘贴的文本.txt`

## 最新实验进度（2026-10-07）

### BGE-large 多查询、融合和重排

- 已完成 BGE-large 规则模板多查询实验，脚本为 exp/bge_large_template_multiquery.py。模板从原问题抽取核心主题/实体、时间范围或锚点、事件/变化/原因/影响及证据类型。
- 多查询结果没有稳定提升：train 的 Top-100/Top-300 下降，dev 的 Top-500 只有小幅提升，说明当前模板查询与原问题相似，新增召回路径有限。
- 已完成 BGE-large 与 BM25 的 Top-100 加权融合比较，测试 BM25 权重 0.1、0.15、0.2、0.25、0.3、0.4、0.5、1.0。
- 按 train 选参，当前推荐 BGE-large:BM25 = 1:0.3；该比例的宏平均 nDCG@10 约为 0.2146（train），dev 约为 0.2873。
- 1:0.3 融合的已标注正例召回率@100约为 train 0.5919、dev 0.5917。BM25 的增益有限，但说明词法检索可以补充少量 BGE 漏掉的候选。
- 已完成 BGE-large + BM25(0.3) Top-100 -> BAAI/bge-reranker-large -> Top-10。重排后 train nDCG@10 从约 0.2146 降至 0.1673，dev 从约 0.2873 降至 0.1926。当前不建议直接使用该通用 reranker；它没有显式建模 TEMPO 的时间匹配、多时期覆盖和时间错误难负例。
- 已完成 BGE 候选深度诊断脚本 bge_recall_diagnostic.py，用于按查询比较 Top-10、Top-100、Top-1000 的正例召回及时间覆盖，区分“正例在更深位置”和“检索器完全找不到”。

### BM25 Top-300 / Top-500 候选生成

- 已创建脚本：exp/bm25_depth_1a.py。
- 参数沿用官方纯 Python BM25：k1=0.9、b=0.4。
- 该脚本为 13 个领域的 train/dev 生成 Top-300 和 Top-500，并检查新文件前 100 名是否与既有 BM25 Top-100 一致。
- 输出目录：
  - exp/runs_bm25_top300/
  - exp/runs_bm25_top500/
- 最近服务器环境：
  - 项目：/home/huangliang/reteco/RETECO
  - 数据：/home/huangliang/reteco/reteco_data
  - Conda Python：/home/huangliang/.conda/envs/reteco/bin/python
- 最近检查时（2026-10-07 13:16:15 UTC）任务仍在运行：
  - PID：124980
  - 命令：python -u exp/bm25_depth_1a.py --data ../reteco_data --domain hsm
  - 后台启动队列：bash exp/run_bm25_depth_1a.sh history genealogy politics hsm economics travel
  - 已完成 Top-300 文件 21/26，Top-500 文件 21/26；另一组 bitcoin cardano law monero quant workplace 已完成。
- 完成标准是两个目录各有 26 个 bm25_1a_*.txt 文件。完成后应先做格式检查，再使用 scorer.py 统计 nDCG@10、TP@10、TR@10、TC@10、NDCG|FC@10、完整覆盖数量和已标注正例召回率。

检查服务器任务：

cd ~/reteco/RETECO
ps -ef | grep -E '[b]m25_depth_1a.py|[r]un_bm25_depth_1a.sh'
find exp/runs_bm25_top300 -maxdepth 1 -name 'bm25_1a_*.txt' | wc -l
find exp/runs_bm25_top500 -maxdepth 1 -name 'bm25_1a_*.txt' | wc -l
tail -f exp/bm25_depth_group0.log

### 当前研究判断和下一步

- 当前最强单一召回器仍是 BGE-large；BM25 主要提供互补词法候选，E5-Mistral 当前结果异常偏低，仍应优先完成 E5 实现审计。
- 首轮正例召回率@100约为 0.5~0.6，意味着候选池中仍可能缺失大量已标注正例；因此在训练 reranker 前，应先比较 Top-100、Top-300、Top-500 的候选召回和时间覆盖。
- BM25 Top-300/500 完成后，优先做：
  1. BM25 与 BGE-large 的 Top-300/500 RRF 或归一化融合；
  2. 比较候选池深度对正例召回、TC 和 nDCG 的影响；
  3. 在扩大后的候选池上加入时间、实体、事件和时期覆盖特征；
  4. 再训练或微调面向 TEMPO 难负例的监督式 reranker。
- 训练和比例选择仍以 train 为准，dev 只用于留出验证；不能因为 dev 最优就把参数当成正式结论。

## Qwen3-Embedding-8B 实验（2026-10-10）

### 任务目标

- 使用 `Qwen/Qwen3-Embedding-8B` 跑 Track 1a 的全部 13 个领域。
- 同时处理 `train` 和 `dev`。
- 每个查询输出 Top-500 候选文档。
- 先用 `iota` 做小规模 smoke test，再运行全部领域，避免模型或 pooling 实现错误在全量任务结束后才发现。

### 脚本和目录

- 本地项目脚本：`H:\code_python\reteco\RETECO\exp\qwen3_embedding_1a.py`
- 服务器脚本：`/home/huangliang/reteco/RETECO/exp/qwen3_embedding_1a.py`
- 模型本地下载目录：`/home/huangliang/models/Qwen3-Embedding-8B`
- 文档向量缓存：`/home/huangliang/reteco/RETECO/exp/cache_qwen3_embedding_8b`
- Top-500 run 输出：`/home/huangliang/reteco/RETECO/exp/runs_qwen3_embedding_8b_top500`
- 运行日志目录：`/home/huangliang/reteco/RETECO/exp/logs_qwen3_embedding_8b`
- 主日志：`/home/huangliang/reteco/RETECO/exp/qwen3_embedding_1a_top500.log`

### 实现配置

- 查询格式：`Instruct: <query instruction>\nQuery: <原始问题>`。
- 默认 instruction：要求检索能够覆盖主题、实体、事件、时间时期、变化、原因、影响和证据类型的段落。
- 文档不添加 query instruction。
- 使用 Qwen 风格的最后一个非 padding token pooling。
- 查询和文档向量进行 L2 归一化，再用点积计算余弦相似度。
- 初始安全参数：`batch=1`、`query-batch=1`、`doc-chunk=1024/2048`、`max-length=2048`、`dtype=float16`。
- 使用两张 3090 时，由主进程通过 `CUDA_VISIBLE_DEVICES` 把不同领域分配给不同 GPU；每个 worker 内部使用可见设备 `cuda:0`。
- 缓存带有模型名、最大长度、文档数量、维度、pooling、归一化和 dtype 元数据，禁止混用 BGE/E5/Qwen 缓存。

### 当前状态

- Qwen3-Embedding-8B 模型已经成功下载到：`/home/huangliang/models/Qwen3-Embedding-8B`。
- 之前的下载失败原因是 Hugging Face Xet CAS 返回 `401 Unauthorized`；已改用镜像并禁用 Xet 后完成下载。
- `transformers` 的弃用警告 `torch_dtype is deprecated` 已在脚本中改为 `dtype=`。
- GPU worker 已修正为继承主进程的 `--splits` 和 `--query-instruction` 参数。
- 在模型下载成功前，iota smoke test 曾在模型加载阶段退出；那次没有生成有效 Qwen run，不能作为模型效果结果。
- 模型下载成功后，需要重新运行 iota smoke test，确认生成的 run 通过格式检查和 scorer，再启动 13 个领域全量任务。
- 截至本次记录，尚未确认全量 Qwen train/dev run 和评分结果已经生成；不要把 Qwen 的性能与 BGE/E5 比较，直到 run 和评分完成。
- Qwen Top-500 评分后发现 bitcoin train/dev nDCG、TP、TR、TC 全为 0；原因不是 bitcoin 没有正例，而是 `float16` 文档缓存有 18 行 NaN 向量。
- 这些 NaN 文档进入所有查询的前排，run 分数列也写成 `nan`；已复测同一文档，`bfloat16` 编码无 NaN。
- Qwen 脚本已按 dtype 区分缓存文件名和元数据；修复后应使用 `--dtype bfloat16` 重新生成 bitcoin（或全量）run，不得复用旧的 float16 bitcoin 缓存。

### 下载和运行环境

服务器运行前设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
```

模型已下载成功后，运行脚本时应通过本地路径加载：

```bash
python -u exp/run_qwen3_embedding_1a.py \
  --data ../reteco_data \
  --model /home/huangliang/models/Qwen3-Embedding-8B \
  --domains iota \
  --gpus 0 \
  --splits train dev \
  --top-k 500 \
  --batch 1 \
  --query-batch 1 \
  --doc-chunk 1024 \
  --max-length 2048 \
  --dtype float16
```

验证 iota 成功后，再使用 `--gpus 0 1`、全部领域和 `--doc-chunk 2048` 运行全量任务。完成后必须对每个领域的 train/dev run 执行 `starter_kit/format_checker.py` 和 `starter_kit/scorer.py`，记录 nDCG@10、TP@10、TR@10、TC@10、NDCG|FC@10、完整覆盖数量和 Top-500 正例召回率。

