# Runnable 与 LCEL 综合实践

## 这节在讲什么

前面已经学习了 LCEL 中的 `Runnable` API，这一节是把它综合用起来：用 `Runnable` 的方式重写之前 MCP、RAG 相关的案例代码。

核心目标不是单独记某个 API，而是理解一种写法：把复杂流程拆成可组合的节点，再用 LCEL 把这些节点声明式地串起来。

## Runnable 的基本流程

使用 `Runnable` 重写业务流程时，可以按下面的顺序思考：

1. 分析整体流程，先明确输入、输出和中间步骤。
2. 把流程拆成多个原子步骤，每一步只负责一件事。
3. 根据步骤之间的关系，选择合适的 `Runnable` API 进行组合。
4. 统一通过 `invoke`、`stream`、`batch` 等方法调用整个 chain。
5. 在 chain 外层或节点上补充配置、重试、兜底、回调等增强逻辑。

这样写的好处是：业务流程会变成一个清晰的执行链，而不是散落在很多临时代码里的过程式逻辑。

## 核心概念

### LCEL

LCEL 可以理解为 LangChain 中组合链路的表达方式。它让模型、Prompt、工具、解析器、检索器等节点都可以变成可组合的组件。

### Runnable

`Runnable` 是 LCEL 的核心抽象。只要一个节点符合 `Runnable` 的接口，它就可以和其他节点组合，形成更大的 chain。

这种思路的关键是：不要只把模型调用看成一次函数调用，而是把整个 AI 应用流程看成多个可编排、可复用、可增强的执行节点。

## 常用调用方式

### invoke

`invoke` 用来执行一次普通调用，适合输入一个请求并等待完整结果返回。

### stream

`stream` 用来流式返回结果，适合模型输出、工具调用过程或长文本生成等需要逐步展示的场景。

### batch

`batch` 用来批量处理多个输入，适合对一组数据执行同一条 chain。

## Chain 的增强能力

写好 chain 之后，可以在不破坏主流程的情况下继续增加一些能力。

### withConfig

`withConfig` 可以给 chain 或某个节点加入配置。chain 中的节点可以通过第二个参数拿到这些配置，用来控制运行时行为。

### withRetry

`withRetry` 用来增加重试逻辑。适合处理模型调用、网络请求、工具调用等可能临时失败的步骤。

### withFallback

`withFallback` 用来增加备选方案。当主 chain 或主节点失败时，可以切换到备用 chain，提升整体稳定性。

### callbacks

`callbacks` 可以给执行过程加回调函数，例如打印某个节点的输入输出、观察执行过程、记录日志等。

## 为什么重要

LCEL 是 LangChain 的重要基础。通过 `Runnable`，不同节点可以被组件化，并且能用统一的方式组合、调用和增强。

后面学习的 LangGraph、LangSmith 也都和 `Runnable` 思想有关：

- LangGraph 会把复杂 Agent 流程组织成图结构，但节点执行仍然离不开类似 `Runnable` 的组合思想。
- LangSmith 会关注 chain 的运行过程、调试、追踪和评估，也需要理解 chain 是怎么被拆分和执行的。

所以这一节要重点熟悉声明式链式写法，而不是只记几个方法名。

## 易错点

- `Runnable` 容易被误解成单个 API，其实它更像是一套统一执行协议。
- 拆分步骤时不要太粗，否则 chain 仍然不清晰；也不要太碎，否则会增加理解成本。
- `withRetry` 适合临时失败，不适合掩盖业务逻辑错误。
- `withFallback` 是兜底方案，不应该替代对主流程错误的排查。
- 使用 `callbacks` 打印日志时，要注意不要泄露敏感输入、密钥或用户隐私。

## 复习问题

1. 为什么 LCEL 要把不同节点都抽象成 `Runnable`？
2. `invoke`、`stream`、`batch` 分别适合什么场景？
3. 如何判断一个流程应该拆成哪些原子步骤？
4. `withRetry` 和 `withFallback` 的区别是什么？
5. 为什么说 LangGraph、LangSmith 的学习也依赖对 `Runnable` 的理解？
