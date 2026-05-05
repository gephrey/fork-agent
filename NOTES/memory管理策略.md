# Memory 管理策略

## 这章在讲什么

Memory 解决的是多轮对话中的“上下文怎么保留”的问题。模型本身不会自动记住历史，每次调用都需要把相关信息重新放进 prompt 或消息数组里。

在 LangChain 里，常见做法是先用 `ChatMessageHistory` 保存消息，再根据上下文窗口、成本和业务目标选择不同的 memory 管理策略。这个仓库里的 `memory-test` 主要演示了三类策略：截断、总结、检索。

## 核心概念

### ChatMessageHistory

`ChatMessageHistory` 是对话消息的存储抽象，用来保存 `HumanMessage`、`AIMessage` 等消息。

常见实现包括：

- `InMemoryChatMessageHistory`：保存在内存里，适合 demo、单进程临时会话。
- Redis / FileSystem / TypeORM 等历史存储：适合跨请求、跨进程或持久化场景。

它只负责“存消息”，不等于已经有了完整的 memory 策略。真正的策略在于：调用模型前，到底取哪些历史、压缩哪些历史、检索哪些历史。

### RunnableWithMessageHistory

`RunnableWithMessageHistory` 可以把消息历史接入 LCEL chain。

关键点：

- `getMessageHistory(sessionId)`：根据会话 ID 找到对应历史。
- `inputMessagesKey`：本轮用户输入对应的字段。
- `historyMessagesKey`：prompt 里 `MessagesPlaceholder` 对应的历史字段。
- `configurable.sessionId`：调用 chain 时传入，用来区分不同用户或不同会话。

这种写法适合简单多轮对话：每次调用时自动读取历史，并在调用结束后写回新的用户消息和 AI 回复。

## 三种常见策略

### 1. 截断策略

截断是最直接的 memory 管理方式：历史过长时，只保留最近一部分消息。

常见截断方式：

- 按消息数量截断：例如只保留最近 4 条消息。
- 按 token 数量截断：例如使用 `trimMessages` 和 `js-tiktoken`，保证历史不超过指定 token 上限。

适合场景：

- 最近上下文最重要的短期对话。
- 对历史准确性要求不高，只需要延续当前话题。
- 想控制 token 成本和响应延迟。

缺点：

- 早期重要信息会被直接丢掉。
- 用户隔了很多轮再追问旧信息时，模型可能答不上来。
- 只按数量截断不够精确，因为不同消息的 token 长度差异很大。

### 2. 总结策略

总结策略是在历史超过阈值时，把较早的消息交给模型生成摘要，然后保留摘要和最近几轮消息。

基本流程：

1. 从 `ChatMessageHistory` 里取出全部消息。
2. 判断是否超过阈值，例如超过 6 条消息。
3. 保留最近几条原始消息。
4. 对更早的消息调用模型生成 summary。
5. 后续 prompt 中带上 summary 和最近消息。

适合场景：

- 对话持续时间较长，但需要保留整体脉络。
- 用户会逐步补充需求、偏好、限制条件。
- 允许把细节压缩成摘要，不要求逐字还原历史。

缺点：

- 摘要本身可能遗漏细节。
- 摘要会引入额外模型调用成本。
- 如果摘要更新逻辑设计不好，错误信息会被不断继承。

### 3. 检索策略

检索策略把历史对话向量化后存入向量数据库，当前用户输入也向量化，再根据语义相似度找回相关历史。

这个仓库中的 `retrieval-memory.mjs` 使用了：

- `OpenAIEmbeddings` 生成向量。
- `MilvusClient` 存储和检索历史对话。
- `MetricType.COSINE` 做余弦相似度匹配。
- 当前问题命中相关历史后，把检索结果拼进 prompt 再调用模型。

基本流程：

1. 用户输入新问题。
2. 对当前问题生成 embedding。
3. 到 Milvus 里检索相似历史。
4. 把相关历史组织成上下文。
5. 调用模型生成回答。
6. 把本轮对话继续写入消息历史和向量库。

适合场景：

- 长时记忆。
- 用户过了很多轮以后追问旧内容。
- 需要从大量历史里找少量相关片段。
- 对话历史可以按语义主题复用。

缺点：

- 需要额外维护向量库。
- 检索质量依赖 embedding 模型、切分方式、metadata 和相似度阈值。
- 语义相似不等于业务相关，可能检索到看似相关但不该使用的历史。

## 实现流程

一个完整的 memory 流程可以按下面理解：

1. 接收用户输入。
2. 根据 `sessionId` 获取对应的 `ChatMessageHistory`。
3. 根据策略构造本轮上下文：
   - 短对话：直接拿最近消息。
   - 长对话：旧消息总结，新消息原样保留。
   - 长时记忆：根据当前 query 检索相关历史。
4. 用 `MessagesPlaceholder` 或手动消息数组把上下文放入 prompt。
5. 调用模型生成回复。
6. 把用户输入和模型回复写回历史。
7. 如果使用检索策略，还要把本轮对话向量化后写入向量库。

## 关键 API / 代码点

### `InMemoryChatMessageHistory`

用于临时保存消息：

```js
const history = new InMemoryChatMessageHistory();
await history.addMessage(new HumanMessage(input));
await history.addMessage(new AIMessage(response));
const messages = await history.getMessages();
```

### `trimMessages`

用于按 token 数量裁剪消息：

```js
const trimmedMessages = await trimMessages(allMessages, {
  maxTokens: 100,
  tokenCounter: async (msgs) => countTokens(msgs, enc),
  strategy: "last",
});
```

`strategy: "last"` 表示优先保留最近消息。

### `getBufferString`

把消息数组转换成适合总结的文本：

```js
const conversationText = getBufferString(messages, {
  humanPrefix: "用户",
  aiPrefix: "助手",
});
```

### `MessagesPlaceholder`

在 prompt 中预留历史消息位置：

```js
const prompt = ChatPromptTemplate.fromMessages([
  ["system", "你是一个简洁、有帮助的中文助手。"],
  new MessagesPlaceholder("history"),
  ["human", "{question}"],
]);
```

### `RunnableWithMessageHistory`

把 chain 和消息历史绑定：

```js
const chain = new RunnableWithMessageHistory({
  runnable: simpleChain,
  getMessageHistory: (sessionId) => getMessageHistory(sessionId),
  inputMessagesKey: "question",
  historyMessagesKey: "history",
});
```

## 策略选择

### 只需要短期上下文

优先用截断策略。实现简单，成本低，适合客服、问答、临时任务助手等场景。

### 需要保留整体脉络

优先用总结策略。它能把长对话压缩成稳定上下文，但要注意摘要质量和更新频率。

### 需要长期个性化记忆

优先用检索策略。用户偏好、项目背景、历史决策、长期计划等信息更适合存到向量库，通过 query 动态召回。

### 实际项目里的组合方式

真实应用通常不是三选一，而是组合：

- 最近 N 轮消息原样保留。
- 更早的会话维护一份滚动 summary。
- 重要事实和长期偏好写入向量库。
- 当前问题先检索长期记忆，再和短期历史一起放进 prompt。

这样可以同时兼顾连贯性、成本和长期记忆能力。

## 易错点

- 不要把 `ChatMessageHistory` 等同于 memory 策略，它只是存储容器。
- 不要无脑把所有历史都塞进 prompt，容易超上下文、成本高、噪声大。
- 截断策略要尽量按 token 控制，单纯按消息数量不稳定。
- 总结策略要保留最近原始消息，否则模型容易丢掉当前话题细节。
- 检索策略要保存 metadata，例如 `sessionId`、时间、轮次、主题，否则多用户或多会话时容易串记忆。
- 长期记忆要注意隐私和删除机制，不是所有用户输入都应该永久保存。
- 工具调用结果、系统消息、用户消息最好分清角色，避免把临时工具输出误当成长期事实。

## 复习问题

1. `ChatMessageHistory` 和 memory 管理策略有什么区别？
2. 截断策略为什么更适合短期上下文？
3. 总结策略为什么需要保留最近几轮原始消息？
4. 检索策略为什么适合长时记忆？
5. `MessagesPlaceholder` 在多轮对话 prompt 中解决了什么问题？
6. 实际项目里为什么常常要组合截断、总结和检索？
