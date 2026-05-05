import 'dotenv/config';
import { ChatOpenAI } from '@langchain/openai';
import { HumanMessage, SystemMessage, ToolMessage } from '@langchain/core/messages';
import { InMemoryChatMessageHistory } from '@langchain/core/chat_history';
import { JsonOutputToolsParser } from '@langchain/core/output_parsers/openai_tools';
import {
  executeCommandTool,
  listDirectoryTool,
  readFileTool,
  writeFileTool,
} from './all-tools.mjs';
import chalk from 'chalk';

const model = new ChatOpenAI({
  modelName: 'qwen-plus',
  apiKey: process.env.OPENAI_API_KEY,
  temperature: 0,
  configuration: {
    baseURL: process.env.OPENAI_BASE_URL,
  },
});

const tools = [
  // 读取项目文件内容，供模型理解当前代码。
  readFileTool,
  // 写入或覆盖文件，供模型直接落地代码修改。
  writeFileTool,
  // 执行终端命令，例如安装依赖、运行测试、启动服务。
  executeCommandTool,
  // 列出目录内容，供模型探索项目结构。
  listDirectoryTool,
];

// 绑定工具到模型
const modelWithTools = model.bindTools(tools);

// Agent 执行函数
async function runAgentWithTools(query, maxIterations = 30) {
  const history = new InMemoryChatMessageHistory();

  await history.addMessage(
    new SystemMessage(`你是一个项目管理助手，使用工具完成任务。

当前工作目录: ${process.cwd()}

工具：
1. read_file: 读取文件
2. write_file: 写入文件
3. execute_command: 执行命令（支持 workingDirectory 参数）
4. list_directory: 列出目录

重要规则 - execute_command：
- workingDirectory 参数会自动切换到指定目录
- 当使用 workingDirectory 时，绝对不要在 command 中使用 cd
- 错误示例: { command: "cd svelte-moodboard-app && pnpm install", workingDirectory: "svelte-moodboard-app" }
- 正确示例: { command: "pnpm install", workingDirectory: "svelte-moodboard-app" }

重要规则 - write_file：
- 当写入 Svelte 组件文件（如 App.svelte）时，优先使用组件内的 <script>、<style> 和模板结构完成交互与样式
`),
  );

  await history.addMessage(new HumanMessage(query));

  for (let i = 0; i < maxIterations; i++) {
    console.log(chalk.bgGreen(`⏳ 正在等待 AI 思考...`));

    // 获取当前消息历史
    const messages = await history.getMessages();

    const rawStream = await modelWithTools.stream(messages);

    // 准备一个空的容器来拼接完整的 AIMessage
    let fullAIMessage = null;

    // 准备一个 tool_call_chunks 的 JSON 增量解析器
    const toolParser = new JsonOutputToolsParser();

    // 记录每个工具调用已打印的长度（用 id 或 filePath 作为 key）
    const printedLengths = new Map();

    console.log(chalk.bgBlue(`\n🚀 Agent 开始思考并生成流...\n`));

    for await (const chunk of rawStream) {
      // 这里的 chunk 是 AIMessageChunk，把它拼接起来
      fullAIMessage = fullAIMessage ? fullAIMessage.concat(chunk) : chunk;

      let parsedTools = null;
      try {
        parsedTools = await toolParser.parseResult([{ message: fullAIMessage }]);
      } catch (e) {
        // 解析失败说明 JSON 还不完整，忽略错误继续累积
      }

      if (parsedTools && parsedTools.length > 0) {
        for (const toolCall of parsedTools) {
          if (toolCall.type === 'write_file' && toolCall.args?.content) {
            const toolCallId = toolCall.id || toolCall.args.filePath || 'default';
            const currentContent = String(toolCall.args.content);
            const previousLength = printedLengths.get(toolCallId);

            if (previousLength === undefined) {
              printedLengths.set(toolCallId, 0);
              console.log(
                chalk.bgBlue(
                  `\n[工具调用] write_file("${toolCall.args.filePath}") - 开始写入（流式预览）\n`,
                ),
              );
            }

            if (currentContent.length > previousLength) {
              const newContent = currentContent.slice(previousLength);
              process.stdout.write(newContent);
              printedLengths.set(toolCallId, currentContent.length);
            }
          }
        }
      } else {
        // 当前还没有解析出工具调用时，如果有文本内容就直接输出
        if (chunk.content) {
          process.stdout.write(
            typeof chunk.content === 'string' ? chunk.content : JSON.stringify(chunk.content),
          );
        }
      }
    }

    // 此时 fullAIMessage 已经完美还原，直接存入 history
    await history.addMessage(fullAIMessage);
    console.log(chalk.green('\n✅ 消息已完整存入历史'));

    // 检查是否有工具调用
    if (!fullAIMessage.tool_calls || fullAIMessage.tool_calls.length === 0) {
      console.log(`\n✨ AI 最终回复:\n${fullAIMessage.content}\n`);
      return fullAIMessage.content;
    }

    // 执行工具调用
    for (const toolCall of fullAIMessage.tool_calls) {
      const foundTool = tools.find((t) => t.name === toolCall.name);
      if (foundTool) {
        const toolResult = await foundTool.invoke(toolCall.args);
        await history.addMessage(
          new ToolMessage({
            content: toolResult,
            tool_call_id: toolCall.id,
          }),
        );
      }
    }
  }

  throw new Error(`Agent reached max
    iterations (${maxIterations}) without
    final response`);
}

const case1 = `创建一个功能丰富的 Svelte AI Moodboard 应用：

1. 创建项目：echo -e "n\nn" | pnpm create vite svelte-moodboard-app --template svelte-ts
2. 修改 src/App.svelte，实现一个新潮的 AI Moodboard 生成器：
 - 输入主题关键词、选择氛围风格、选择配色强度
 - 一键生成 6 张灵感卡片，每张包含标题、提示词、颜色标签、热度分数
 - 支持收藏/取消收藏灵感卡片
 - 支持按全部/收藏筛选
 - 支持随机刷新单张卡片
 - localStorage 持久化主题、收藏和生成结果
3. 添加精致样式：
 - 玻璃拟态面板、深色未来感背景
 - 霓虹描边、柔和光效、响应式网格
 - 按钮、输入框、卡片都要有清晰的交互状态
4. 添加动画：
 - 卡片生成时的入场动画
 - 收藏和刷新时使用 CSS transitions
5. 列出目录确认

注意：使用 pnpm，功能要完整，样式要美观，要有动画效果

去掉 src/main.ts 里的 app.css 导入（如果存在）

之后在 svelte-moodboard-app 项目中：
1. 使用 pnpm install 安装依赖
2. 使用 pnpm run dev 启动服务器
`;

try {
  await runAgentWithTools(case1);
} catch (error) {
  console.error(`\n❌ 错误: ${error.message}\n`);
}
