# RAG 离线评估：人工金标注流程与设计思路

本文说明**为何要做人工金标**、**标注什么**、**按什么顺序做**、**如何质检与维护**，并与本仓库 `rag_eval_queries.jsonl`、`search_chunks`、`eval_retrieval` 对齐。

---

## 一、设计目标与原则

### 1.1 要解决什么问题

- **检索评估**需要「-query → 哪些 chunk 算对」的**金标准**，才能计算 P@k、R@k、MRR、nDCG 等。
- 若金标来自“自动候选池回写”（例如三路候选并集），属于弱监督口径，可能稀释策略差异；严肃对比建议人工金标。
- **人工金标**的目标是：在**固定语料与索引版本**下，由人定义「哪些 passage 足以支撑回答该问题」，使指标反映**真实检索质量**，而非「和 baseline 有多像」。

### 1.2 设计原则（建议写进标注规范）

| 原则 | 说明 |
|------|------|
| **可复现** | 金标绑定**索引版本**（数据、切分、embedding 模型）；换库或重建后需重新对齐 id。 |
| **任务一致** | 先明确评估的是「**段落是否相关**」还是「**文档是否相关**」；本仓库默认以 **chunk id** 为主，URL 由 chunk 元数据推导，属较粗粒度。 |
| **可操作** | 每条 query 的 `relevant_ids` 应是**当前向量库中可查**的 UUID；不可用「想象」的 id。 |
| **可审计** | 保留 `gold_answer`（可选）与**简短标注说明**（见下文「标注记录表」），便于复核与争议处理。 |

---

## 二、整体流程（鸟瞰）

```text
确定评估范围与指标定义
        ↓
冻结语料与索引（版本记录）
        ↓
编写/整理 query 列表（与业务问题一致）
        ↓
编写《标注规范》（1 页以内）
        ↓
标注员：用 search_chunks 检索 → 人工判定 → 写入 relevant_ids
        ↓
自检：run eval_retrieval + 校验 id 存在性
        ↓
（可选）交叉复核 / 仲裁
        ↓
定稿 jsonl + 记录版本 + 跑完整基线实验
```

### 会话脚本（按顺序执行）

在**项目根目录**、已 `pip install -e .` 或已配置 `PYTHONPATH` 的前提下：

| 步骤 | 命令 / 动作 | 说明 |
|------|----------------|------|
| 1 | `python -m src.evaluator.rag_eval_runner gold_candidates` | 仅展示 **baseline** **Top-k**（默认 10）。**不要**把「只从这里面勾选」当作公平金标（见下文「方法学」）。 |
| 1b | `python -m src.evaluator.rag_eval_runner gold_pool` | **baseline + HyDE + MQE** 三路各 Top-k，**去重合并**为更宽候选池；HyDE/MQE 单独召回到的 chunk 也会出现，减轻「金标只含 baseline 窗口」对召回类方法的偏见。需 LLM，较慢。 |
| 2 | （可选）`python -m src.evaluator.rag_eval_runner search_chunks "补充关键词"` | 换中英文关键词在全库检索；**真正公平**还需靠你对 `gold_answer`/题意的判断，把**任意**相关 chunk id 写入 `relevant_ids`（不限于某一路 Top-k）。 |

### 方法学：为什么「只从 baseline 的 10 条里选」不公平？

- `gold_candidates` 里的列表来自 **Base RAG 的向量检索**，**不是**「全库里所有应算相关的文档」。
- 若 `relevant_ids` **只**从这 10 条里选，等于默认：**所有正确答案都已经落在 baseline 的 Top-k 里**；任何 **baseline 没排进 Top-k、但被 HyDE/MQE 召回到的相关文档** 都不会出现在金标集合里。
- 后果：在 **Recall@k** 等指标上，**提高召回** 的改进（HyDE、MQE、换 embedding 等）会被**系统性低估**——它们多召回来的「新」相关文档，你的金标里根本没有，Recall 无法体现。
- **缓解（仍非全库完美）**：用 **`gold_pool`** 扩大候选面；用 **`search_chunks`** + **`dump_chunk`** 按关键词/全文在全库补标；或采用文献中的 **pooling / 多标注员 / 分层抽样** 等更重工序。
- **自动候选池回写** 与「只从某一路候选里勾选金标」都可能引入偏置；公平对比 HyDE/MQE 时建议使用 `gold_pool` 扩大候选并进行人工标注/复核。
| 3 | 编辑 `data/rag_eval_queries.jsonl` | 把人工认可的 UUID 写入 `relevant_ids`；**不要**用 `gold_generated_by: baseline_topk` 表示人工金标（可删除该字段） |
| 4 | `set RAG_EVAL_MODES=baseline` 后 `python -m src.evaluator.rag_eval_runner eval_retrieval` | 快速自检：金标 id 是否存在、chunk 级指标是否合理 |
| 5 | 取消仅 baseline，再跑 `eval_retrieval` | 对比 HyDE/MQE（需 LLM，较慢） |

环境变量：`RAG_EVAL_QUERIES_PATH` 可指向非默认 jsonl。

### 常见问题：为什么 preview 只有一段？URL 是 N/A？

- **preview 不完整**：`gold_candidates` / `search_chunks` 里打印的 `preview=` 只是**控制台预览**，默认截断为前 **220 个字符**（避免刷屏），**不是**库里只存了这一段。需要看更长可把环境变量 **`RAG_EVAL_GOLD_PREVIEW_CHARS`** 调大（例如 `500`）。
- **url=N/A**：评估索引里 chunk 的 `source_urls` 来自建索引时的元数据。若 HTML 清洗后写了 `SourceURL:` 行，**新版本**会把该 URL 写入**同一文件切出的所有 chunk**；若仍是 N/A，可能是纯 `.txt` 无溯源行、或索引在修复前已建好——需 **`build_index` 重建**（并保留 `xxx.html.url.txt` 等溯源文件）后才会在候选列表里看到 URL。
- **preview 里还有「复制页面」、`*]:mt-3">` 等**：来自网页导航/样式未剥干净；`build_index` 已对 HTML 做导航区去除与噪声短语清理。若仍看到旧噪声，请**重新执行 `build_index` 重建向量库**后再用 `gold_candidates`。

---

## 三、各阶段详细步骤

### 阶段 A：评估范围与指标定义（第 0 步）

**目的**：避免后续「标了但和指标语义不一致」。

1. **明确任务**  
   - 例：仅评估「**检索**」是否召回**能回答该问题的段落**（不评估生成质量）。  
   - 若还要评估生成，可另设 `gold_answer` 或走 `eval_langsmith` 等链路。

2. **明确 k**  
   - 与 `eval_retrieval` 的 **k**（默认 5）一致；若你关心 P@3，则标注时心里也要有「**前 3 个里应出现哪些**」的预期。

3. **明确粒度**  
   - **推荐**：以 **chunk（段落）** 为相关单元；**不要用 URL 当唯一金标**（颗粒过大，见 README 讨论）。  
   - 本仓库：jsonl 里写 **`relevant_ids`**；若 chunk 带 `source_urls`，评估会自动用 URL 级指标作为**补充**，但**主结论仍建议以 chunk 级为准**。

4. **输出物**  
   - 一页「评估说明」：目标、k、粒度、是否含 `gold_answer`。

---

### 阶段 B：冻结语料与索引（版本化）

**目的**：chunk id 与向量库内容一一对应；**换库后 id 会变**。

1. **固定**  
   - 数据目录、切分脚本、`build_index` 使用的 **research_topic**（如 `rag_eval`）。  
   - 记录：**embedding 模型版本**、Chroma 路径、构建时间。

2. **构建索引**（若尚未构建）  
   ```bat
   python -m src.evaluator.rag_eval_runner build_index
   ```

3. **记录**  
   - 在团队内或 `README` 旁维护一句：**「金标基于某次 build 的 DB」**；大改数据后**必须**重标；若仅为快速对齐 chunk_id，可用 `refresh_eval_ids_union` 生成弱监督候选并再人工修正。

---

### 阶段 C：准备 query 列表

**目的**：query 应代表真实用户或业务场景，而不是「为了凑检索」而写。

1. **来源**  
   - 真实日志脱敏、产品 FAQ、考试/教材章节问题等。  
   - 每条 query **表述清晰、无歧义**（避免「它」「这个」无上下文）。

2. **规模**  
   - 起步可 **10～30 条** 做方法对比；扩大集前先把规范与流程跑通。

3. **写入 jsonl 骨架**（可先无 `relevant_ids` 或空列表，后续再填）  
   - 字段：`id`, `query`, `relevant_ids`, （可选）`gold_answer`。  
   - **不要**在人工定稿前使用 `gold_generated_by: baseline_topk`，以免误导读者以为仍是 baseline 金标。

---

### 阶段 D：编写《标注规范》（强烈建议）

**目的**：统一「什么叫相关」，降低不同人标注漂移。

建议至少包含：

1. **相关（positive）**  
   - 该 chunk 的**正文**直接包含可回答问题的**事实、定义、步骤或结论**；或  
   - 虽不完整，但与问题**强相关**且可作为**主要引用**之一（团队需统一是否收录「弱相关」）。

2. **不相关（negative）**  
   - 仅同主题但**不能回答该具体问题**；  
   - 仅同关键词但**答非所问**。

3. **边界**  
   - 同一文档多 chunk：若**只有其中一段**真正有用，只标**那些**；**不要为了凑数**把整页所有 chunk 都标上。  
   - **相关条数**：可设上限（如每 query 最多 **5～10 个** id），避免 Recall 被「超大金集」稀释。

4. **冲突处理**  
   - 两人不一致时：以**第三者仲裁**或** majority**。

---

### 阶段 E：单条 query 的标注操作（逐步）

以下假设已装好依赖、可访问评估向量库。

#### E1. 理解问题

- 读 `query`，用一句话写下**期望答案类型**（定义/对比/步骤/列举…）。

#### E2. 检索候选 chunk

使用项目自带命令（**必须用当前评估库**）：

```bat
python -m src.evaluator.rag_eval_runner search_chunks "与问题相关的关键词或短语"
```

- **技巧**：换用**同义词、英文/中文、文档里的专有名词**多搜几次（`k` 默认 10，可在代码里改或通过参数若已暴露）。  
- 记录：候选的 `id`、`url`、`preview` 是否**真相关**。

#### E3. 判定与收集 id

- 对每条候选：**相关** → 复制**完整 UUID** 到列表；**不相关** → 忽略。  
- 若 Top-10 都不相关：**扩大关键词**或检查**是否索引里根本没有该内容**（需回到语料/索引，而不是硬标）。

#### E4. 写入 `rag_eval_queries.jsonl`

- 对应 `id` 的那一行，把 `relevant_ids` 设为 UUID 字符串数组。  
- JSON 格式：一行一个对象，**合法 JSON**（引号、逗号、无尾逗号）。

示例：

```json
{"id": "q1", "query": "……", "relevant_ids": ["uuid-1", "uuid-2"], "gold_answer": "（可选）参考答案"}
```

#### E5. （可选）`gold_answer`

- 用于后续 QA 或人工对照；**不参与**当前 `eval_retrieval` 的 P/R/MRR/nDCG（除非你们扩展代码）。

---

### 阶段 F：自检与自动校验

1. **运行评估**（可先只跑 baseline）  
   ```bat
   set RAG_EVAL_MODES=baseline
   python -m src.evaluator.rag_eval_runner eval_retrieval
   ```

2. **看控制台**  
   - 程序会校验 **金标 id 是否存在于当前 Chroma**；若有缺失，按提示**替换为无效 id** 或重建索引后重搜。  
   - 若指标**全 0**，优先查：**id 是否失效**、**research_topic 是否一致**。

3. **合理性**  
   - 人工金标下，**不应**出现「baseline 永远碾压一切」**仅因**金标来自 baseline 的那种情况；若仍异常，查检索是否故障、或金标是否过宽。

---

### 阶段 G：交叉复核（推荐）

1. **A 标注 → B 抽查**（如 20%）：  
   - 是否漏标、多标、误标。  
2. **记录**  
   - 争议 query 与**最终裁定**写入表格（见下一节）。

---

### 阶段 H：定稿与版本管理

1. **备份**  
   - 复制 `rag_eval_queries.jsonl` 为 `rag_eval_queries.v1.jsonl` 或 git 提交。  
2. **元数据**  
   - 在团队文档或 jsonl 同目录 `README_gold.txt` 中记录：**日期、索引版本、标注人、规范版本**。  
3. **实验**  
   - 再跑 `baseline,hyde,mqe` 全量对比，保存日志。

---

## 四、标注记录表（建议模板）

| qid | query 摘要 | 标注人 | 检索用词 | 相关 id 数量 | 备注（是否弱相关/边界） |
|-----|------------|--------|----------|--------------|-------------------------|
| q1  | …          | 张三   | …        | 3            | uuid-2 仅含定义前半句 |

用于**复盘**与**规范迭代**。

---

## 五、常见陷阱与对策

| 现象 | 可能原因 | 对策 |
|------|----------|------|
| 金标 id 大量不存在 | 重建索引后 id 变化 | `search_chunks` 重新搜；或用 `refresh_eval_ids_union` 生成候选并再人工删改 |
| 指标全 0 | 路径错误、topic 不对、或 id 全失效 | 看评估打印的校验与路径 |
| HyDE 永远低于 baseline | 金标曾是 baseline Top-k | 改人工金标；勿用 `gold_generated_by: baseline_topk` 解读 |
| URL 指标虚高 | URL 粒度粗 | **主看 chunk 级**；URL 仅作辅助 |

---

## 六、与仓库命令的对应关系

| 步骤 | 命令 / 文件 |
|------|-------------|
| 建索引 | `python -m src.evaluator.rag_eval_runner build_index` |
| 批量看候选（推荐第一步） | `python -m src.evaluator.rag_eval_runner gold_candidates` |
| 搜 chunk、取 id（单关键词） | `python -m src.evaluator.rag_eval_runner search_chunks "关键词"` |
| 金标数据 | `data/rag_eval_queries.jsonl`（或 `RAG_EVAL_QUERIES_PATH`） |
| 自动对齐 id（弱监督口径） | `python -m src.evaluator.rag_eval_runner refresh_eval_ids_union` |
| 跑检索指标 | `python -m src.evaluator.rag_eval_runner eval_retrieval` |

---

## 七、小结

- **人工金标的核心**：在**固定索引**下，用 **`search_chunks` 找到真实 UUID**，再按**统一规范**写入 `relevant_ids`。  
- **设计思路**：先锁版本与任务定义，再写规范，再标注、自检、复核，最后版本化；**chunk 级**为主，**URL 级**仅作粗粒度理解。  
- 按此流程，指标才能**解释 HyDE/MQE 等方法的差异**，而不是重复「和 baseline 有多像」。
