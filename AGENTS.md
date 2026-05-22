# fork-agent 项目说明

这个仓库是一个面向 AI Agent、LangChain/LangGraph、RAG、工具调用、语音服务和前端算法练习的多项目工作区。根目录下的大多数文件夹都是彼此独立的小实验或示例项目，每个子项目优先按自己的 `package.json`、`README.md` 和本地目录结构理解。

## 顶层组成

- `/FE`：前端/算法刷题笔记。每个文件通常对应一道题，文件名使用编号和英文题名，例如 `001-two-sum.md`。内容以题目、最优思路、关键点、复杂度和 JavaScript 题解为主。
- `/NOTES`：学习笔记区。用于沉淀项目学习过程中的概念、章节总结、代码结构理解和复盘内容。
- `plan.md`：计划区。用于记录近期学习计划、任务、截止时间和状态检查。当前仓库已有 `PLAN.txt`，如果后续迁移为 `plan.md`，语义保持一致。
- `/nestjs-docs`: 是 Nest.js 官方文档的拷贝，一切对 nest.js 相关的知识点提问的解答，都需要遵从这个文档项目提供的回答。

## 主要子项目

- `runnable-test`：Runnable/LCEL 相关练习和案例。
- `memory-test`：对话历史、摘要记忆、检索记忆、截断记忆等 memory 机制实验。
- `tool-test`：LangChain tool、MCP、文件读取、Node 执行等工具调用实验。
- `asr-and-tts-nest-service`：NestJS 语音相关服务，包含 ASR/TTS、AI 控制器和公开测试页面。
- `agui-backend` / `agui-frontend`：AG-UI 相关前后端实验。
- `advanced-rag`、`rag-test`：RAG 相关实验。
- `langgraph-test`：LangGraph 相关实验。
- 其他目录多为独立验证项目，处理时先阅读对应目录内的 `package.json` 和源码入口。

## 工作约定

- 修改某个子项目时，先进入对应目录查看本地脚本和依赖，不要假设根目录有统一构建命令。
- 新增笔记优先放到 `/NOTES`，算法题相关内容优先放到 `/FE`。
- 保持笔记中文为主，代码和 API 名称保留英文。
- 对已有笔记续写时，延续原文件标题、分节和语气，不做无关重排。
- 对已有代码修改时，优先保持子项目原有技术栈、目录结构和命名风格。

## 笔记 skill

项目内有一个用于写笔记的 skill：

```text
/Users/gephrey/fork-agent/.agents/skills/fork-agent-note-writer/SKILL.md
```

这个 skill 的笔记目标目录是：

```text
/Users/gephrey/fork-agent/NOTES
```

用法：

```text
/skill 增加或续写这一章的内容的笔记 /xx/xx/xxx
```

执行含义：

- `/skill` 后面的自然语言描述是本次笔记任务，例如“增加这一章的内容的笔记”或“续写这一章的内容的笔记”。
- 最后的 `/xx/xx/xxx` 是来源路径、章节路径或要参考的本地材料路径。
- 如果目标章节已有对应笔记，优先续写和补全；如果没有，则在 `/NOTES` 下新建清晰命名的 Markdown 文件。
- 笔记应提炼结构、关键概念、实现流程、易错点和后续可复习的问题，避免只做流水账摘抄。


## 回答要求
每次回答结束，都要在回答末端加上[END]