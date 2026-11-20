# SFT 数据生成方法总结（A1 → C4）

> 本文档总结 Chapter 7 notebook 中从 A1 到 C4 的各种 SFT 数据生成思路。

---

## A 路线：原文改写型

原文已有现成内容，直接整理或改写成问答样本。

### A1. Direct Extraction（直接抽取）

**思路**：原文本身已有"题目 + 解答"结构，直接按原文内容回答问题，不引入额外信息。

**适用场景**：原文有明确的题目和答案边界（如例题页）。

**参考**：MAmmoTH2 的 Recall → Extract → Refine 流程。

---

### A2. Backtranslation（反推问题）

**思路**：原文是说明文，没有现成问题。先让模型"反推"读者最可能会问什么，再根据原文回答这个问题。

**适用场景**：原文是定义页、解释页等不带 instruction 结构的说明文。

**参考**：Self-Alignment with Instruction Backtranslation（arxiv 2308.06259）。

---

### A3. Back-and-Forth（改写风格）

**思路**：原文的数学内容正确，但表达太像网页摘录。保留原问题，根据原文把回答重写成更像教材或讲义的风格。

**适用场景**：原文内容对，但风格不适合训练数据。

**参考**：Better Alignment with Instruction Back-and-Forth Translation（EMNLP 2024）。

---

### A4. MIND-style Guidance（教师辅导风格）

**思路**：把"说明为什么"的原文重写成教师辅导口吻，强调讲解过程而不只是给出答案。

**适用场景**：原文有"验证解"、"为什么要检验"等解释性内容。

**参考**：MIND: Math Informed syNthetic Dialogues（OpenReview）。

---

## B 路线：种子扩展型

以现有题目为出发点，扩展出更多同类型样本。

### B1. MetaMath-style（同知识点变式）

**思路**：以一条好题为 seed，做同知识点的变式题，再让模型解答新题。快速扩充题量，同时保持知识点分布稳定。

**参考**：MetaMath。

---

### B2. MathGenie-style（骨架迁移）

**思路**：先从原题中提炼"解法骨架"（如"单价 × 数量 = 总量"），再把骨架迁移到新题上。比简单改数字更能保证同类方法的一致性。

**参考**：MathGenie。

---

### B3. KPDDS-style（关键点驱动）

**思路**：先定位一个核心技能（如"代回检验"），再围绕这个关键点出题并回答。训练目标更精准，直指特定技能。

**参考**：KPDDS（arxiv 2403.02333）。

---

### B4. MathScale / GSDP-style（概念组合）

**思路**：把两个相关概念合并到同一道题中，如"列方程"+"验证"，让一条样本同时训练多个技能点之间的连接关系。

**参考**：MathScale、GSDP（OpenReview）。

---

### B5. Thinking Synthesis for Seed Expansion（种子扩展思维链合成）

**思路**：不要求小模型直接生成完整思维链，而是先保留已生成的可靠答案，再根据题型模板合成一个简短的 reasoning block，最后拼成 `<think>...</think> 正式答案` 格式。

**适用场景**：B 路线生成的普通 SFT 样本升级为 reasoning 格式。

---

## C 路线：合成难化型

对原始题目进行难度增强，或按难度分层组织。

### C1. Evol-Instruct-style（加难指令）

**思路**：在原题基础上增加一个额外要求（如加入检验、比较、解释），使题目更像 instruction-following 任务，难度自然提升。

**参考**：WizardLM / Evol-Instruct。

---

### C2. DART-style（多次采样取优）

**思路**：对较难题目多次采样（如 3 次），再从多个结果中选择更完整、更可靠的一条保存。针对难题单次采样不稳定的问题。

**参考**：DART-Math。

---

### C3. Curriculum / Light-R1-style（难度分层）

**思路**：把样本按 easy / medium / hard 分层组织，而不是全部混在一起。帮助模型从简到难逐步学习，避免难题压倒简单题的分布问题。

**参考**：Light-R1。

---

### C4. Thinking Synthesis for Hard Data（难题思维链合成）

**思路**：与 B5 类似，但针对 C 路线较难的题目，合成更长、更完整的 reasoning block。先保留可靠答案，再为难题补出更细致的 `<think>...</think>` 推理过程。

**适用场景**：C 路线生成的普通 SFT 样本升级为 hard reasoning 格式。

---

## 总览对比

| 方法 | 路线 | 核心操作 | 适用原文类型 |
|------|------|----------|-------------|
| A1 Direct Extraction | A 改写 | 直接抽取原文问答 | 题目+解答页 |
| A2 Backtranslation | A 改写 | 反推问题再回答 | 说明文、定义页 |
| A3 Back-and-Forth | A 改写 | 保留问题，改写回答风格 | 内容正确但风格差的页面 |
| A4 MIND-style | A 改写 | 改写为教师辅导口吻 | 解释性、验证性原文 |
| B1 MetaMath | B 扩展 | 同知识点变式 | 有好题的页面 |
| B2 MathGenie | B 扩展 | 骨架迁移到新题 | 有解法结构的页面 |
| B3 KPDDS | B 扩展 | 关键点驱动出题 | 有核心技能点的页面 |
| B4 MathScale | B 扩展 | 两概念组合成一题 | 多个相关概念的页面 |
| B5 Thinking Synthesis | B 扩展 | 合成短 thinking 块 | B 路线已生成样本 |
| C1 Evol-Instruct | C 难化 | 在原题上加额外要求 | 普通例题 |
| C2 DART | C 难化 | 多次采样取最优 | 较难题目 |
| C3 Curriculum | C 难化 | easy/medium/hard 分层 | 不同难度混合 |
| C4 Thinking Synthesis | C 难化 | 合成长 thinking 块 | C 路线已生成样本 |
