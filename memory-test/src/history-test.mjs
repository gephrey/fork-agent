/**
 * 内存聊天历史演示
 * 
 * 本文件演示了如何使用 LangChain 的 InMemoryChatMessageHistory
 * 在内存中管理多轮对话历史，使 AI 能够基于上下文进行连贯的对话。
 * 
 * 实现流程：
 * 1. 初始化 ChatOpenAI 模型（支持自定义 baseURL）
 * 2. 创建 InMemoryChatMessageHistory 实例作为对话存储
 * 3. 定义系统消息，设置 AI 角色为"友好幽默的做菜助手"
 * 4. 第一轮对话：用户提问 -> 存入历史 -> 调用模型 -> 保存回复
 * 5. 第二轮对话：用户追问 -> 存入历史 -> 携带完整历史调用模型 -> 保存回复
 * 6. 最后遍历 history.getMessages() 展示所有保存的消息记录
 * 
 * 核心要点：通过每次调用时传入 [systemMessage, ...history.getMessages()]
 * 实现对话上下文的累积传递，让 AI 能够记住之前的对话内容。
 */
import 'dotenv/config';
import { ChatOpenAI } from '@langchain/openai';
import { InMemoryChatMessageHistory } from "@langchain/core/chat_history";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

const model = new ChatOpenAI({ 
  modelName: process.env.MODEL_NAME,
  apiKey: process.env.OPENAI_API_KEY,
  temperature: 0,
  configuration: {
      baseURL: process.env.OPENAI_BASE_URL,
  },
});

async function inMemoryDemo() {
  const history = new InMemoryChatMessageHistory();

  const systemMessage = new SystemMessage(
    "你是一个友好、幽默的做菜助手，喜欢分享美食和烹饪技巧。"
  );

  // 第一轮对话
  console.log("[第一轮对话]");
  const userMessage1 = new HumanMessage(
    "你今天吃的什么？"
  );
  await history.addMessage(userMessage1);
  
  const messages1 = [systemMessage, ...(await history.getMessages())];
  const response1 = await model.invoke(messages1);
  await history.addMessage(response1);
  
  console.log(`用户: ${userMessage1.content}`);
  console.log(`助手: ${response1.content}\n`);

  // 第二轮对话（基于历史记录）
  console.log("[第二轮对话 - 基于历史记录]");
  const userMessage2 = new HumanMessage(
    "好吃吗？"
  );
  await history.addMessage(userMessage2);
  
  const messages2 = [systemMessage, ...(await history.getMessages())];
  const response2 = await model.invoke(messages2);
  await history.addMessage(response2);
  
  console.log(`用户: ${userMessage2.content}`);
  console.log(`助手: ${response2.content}\n`);

  // 展示所有历史消息
  console.log("[历史消息记录]");
  const allMessages = await history.getMessages();
  console.log(`共保存了 ${allMessages.length} 条消息：`);
  allMessages.forEach((msg, index) => {
    const type = msg.type;
    const prefix = type === 'human' ? '用户' : '助手';
    console.log(`  ${index + 1}. [${prefix}]: ${msg.content.substring(0, 50)}...`);
  });
}

inMemoryDemo().catch(console.error);