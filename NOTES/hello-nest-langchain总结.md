# hello-nest-langchain 总结

## 这章在讲什么

`hello-nest-langchain` 是一个把 NestJS 和 LangChain 结合起来的小项目。它主要演示：

- Nest 应用如何组织 `Module`、`Controller`、`Service`。
- 如何通过 `ConfigModule` 加载 `.env` 环境变量。
- 如何用 Nest 的 IoC 容器创建并注入 `ChatOpenAI`。
- 如何用 LangChain 的 LCEL 链处理普通问答和流式问答。
- 为什么企业级后端项目更常用 Nest，而不是只用 Express。

这个项目目前没有 RAG、数据库问答、工具调用或记忆机制，核心逻辑是：接收用户问题，包装成 prompt，调用聊天模型，然后返回模型输出。

## 项目结构

关键文件：

- `src/main.ts`：Nest 应用入口，创建 `AppModule` 并监听端口。
- `src/app.module.ts`：根模块，注册 `BookModule`、`AiModule`、静态资源服务和全局配置服务。
- `src/ai/ai.module.ts`：AI 模块，注册 `AiController`、`AiService` 和 `CHAT_MODEL` provider。
- `src/ai/ai.controller.ts`：AI 路由入口，提供普通问答接口和 SSE 流式接口。
- `src/ai/ai.service.ts`：真正组织 LangChain chain 的地方。
- `public/sse-test.html`：浏览器端测试 SSE 流式接口的页面。

模块关系可以理解为：

```text
main.ts
  -> AppModule
      -> ConfigModule
      -> ServeStaticModule
      -> BookModule
      -> AiModule
          -> AiController
          -> AiService
          -> CHAT_MODEL
```

## 普通问答接口流程

访问：

```text
GET http://localhost:3000/ai/chat?query=红烧肉的做法
```

实际流程：

```text
浏览器请求 /ai/chat
  -> AiController.chat()
  -> 读取 query 参数
  -> AiService.runChain(query)
  -> PromptTemplate 生成提示词
  -> ChatOpenAI 调用模型
  -> StringOutputParser 转成字符串
  -> 返回模型结果
```

`AiController` 负责接收 HTTP 请求：

```ts
@Get('chat')
async chat(@Query('query') query: string) {
  const answer = await this.aiService.runChain(query);
  return answer;
}
```

`AiService` 负责调用 LangChain 链：

```ts
async runChain(query: string): Promise<string> {
  return this.chain.invoke({ query });
}
```

这里的 `query` 会被填入 prompt 模板：

```text
请回答以下问题：

红烧肉的做法
```

## LangChain Chain 的组成

`AiService` 里创建了一个 LCEL 链：

```ts
this.chain = prompt.pipe(model).pipe(new StringOutputParser());
```

可以拆成三段：

```text
PromptTemplate
  -> ChatOpenAI
  -> StringOutputParser
```

含义：

- `PromptTemplate`：把用户输入变成完整提示词。
- `ChatOpenAI`：调用 OpenAI 或 OpenAI-compatible 聊天模型接口。
- `StringOutputParser`：把模型返回的消息对象转成普通字符串。

所以这个服务本质上不是自己实现 AI，而是把 Nest 的 HTTP 接口和 LangChain 的模型调用链连接起来。

## ConfigModule 和环境变量

`app.module.ts` 里这段专门用于加载环境变量：

```ts
ConfigModule.forRoot({
  isGlobal: true,
  envFilePath: '.env',
})
```

作用：

- 从项目根目录的 `.env` 文件读取配置。
- 注册全局 `ConfigService`。
- 其他模块不用重复导入 `ConfigModule`，也可以注入 `ConfigService`。

`ai.module.ts` 里通过 `ConfigService` 读取模型配置：

```ts
model: configService.get('MODEL_NAME'),
apiKey: configService.get('OPENAI_API_KEY'),
configuration: {
  baseURL: configService.get('OPENAI_BASE_URL'),
},
```

这些配置决定了：

- 用哪个模型。
- 用哪个 API Key。
- 请求发到哪个 OpenAI-compatible 服务地址。

## IoC 和依赖注入

Nest 相比 Express 的一个重要优势是：Nest 内置 IoC 容器。

IoC 是 `Inversion of Control`，中文叫“控制反转”。大白话是：

```text
以前：我需要什么，我自己 new。
现在：我声明我需要什么，框架帮我传进来。
```

普通写法：

```ts
class AiController {
  private aiService = new AiService();
}
```

这种写法的问题是：`AiController` 自己创建了 `AiService`，两者绑得比较死。

Nest 写法：

```ts
class AiController {
  constructor(private readonly aiService: AiService) {}
}
```

这里 `AiController` 不关心 `AiService` 怎么创建，只声明自己需要一个 `AiService`。创建和注入由 Nest 容器负责。

好处：

- 依赖关系更清楚，看构造函数就知道这个类依赖什么。
- 测试更方便，可以传入 mock service。
- 项目变大后，依赖关系由容器统一管理，不需要到处手写 `new`。

## 构造器注入和属性注入

`AiService` 中使用的是构造器注入：

```ts
constructor(
  @Inject('CHAT_MODEL') model: ChatOpenAI,
) {
  this.chain = prompt.pipe(model).pipe(new StringOutputParser());
}
```

这里适合用构造器注入，因为 `chain` 在构造函数里就要创建，而创建 `chain` 时必须马上拿到 `model`。

如果改成属性注入：

```ts
@Inject('CHAT_MODEL')
private readonly model: ChatOpenAI;

constructor() {
  this.chain = prompt.pipe(this.model).pipe(new StringOutputParser());
}
```

可能会有问题：`constructor` 执行时对象还没完全创建好，Nest 还没有把属性注入进去，所以 `this.model` 可能还是 `undefined`。

记忆方式：

```text
构造器注入：创建对象时依赖就已经传进来了。
属性注入：对象先创建，之后 Nest 再把属性补上。
```

如果某个依赖在构造函数里马上要用，优先使用构造器注入。

## CHAT_MODEL Provider

`ai.module.ts` 里定义了一个自定义 provider：

```ts
{
  provide: 'CHAT_MODEL',
  useFactory: (configService: ConfigService) => {
    return new ChatOpenAI({
      model: configService.get('MODEL_NAME'),
      apiKey: configService.get('OPENAI_API_KEY'),
      configuration: {
        baseURL: configService.get('OPENAI_BASE_URL'),
      },
    });
  },
  inject: [ConfigService],
}
```

这段的意思是：

- 注册一个 token：`CHAT_MODEL`。
- 当其他地方注入 `CHAT_MODEL` 时，Nest 调用 `useFactory` 创建对象。
- `useFactory` 依赖 `ConfigService`，所以通过 `inject: [ConfigService]` 声明依赖。

`ChatOpenAI` 不是所有模型的通用类，它主要适合：

- OpenAI 官方模型。
- 提供 OpenAI-compatible API 的第三方模型服务。

如果是 Anthropic 原生 Claude API，更合适的是 `ChatAnthropic`，而不是 `ChatOpenAI`。如果某个中转平台把 Anthropic 包成 OpenAI-compatible 接口，则可能仍然可以用 `ChatOpenAI`。

## SSE 流式接口

`AiController` 中还有一个流式接口：

```ts
@Sse('chat/stream')
chatStream(@Query('query') query: string): Observable<{ data: string }> {
  return from(this.aiService.streamChain(query)).pipe(
    map((chunk) => ({ data: chunk })),
  );
}
```

这里的 `Observable<{ data: string }>` 表示：这个接口不是一次性返回完整结果，而是持续向浏览器推送数据。

流程：

```text
AiService.streamChain(query)
  -> 返回 AsyncGenerator<string>
  -> from(...) 转成 Observable
  -> map(...) 包装成 { data: chunk }
  -> Nest SSE 持续推给浏览器
```

如果模型逐段返回：

```text
红烧肉
的做法
是先焯水
```

SSE 会依次推送：

```ts
{ data: '红烧肉' }
{ data: '的做法' }
{ data: '是先焯水' }
```

浏览器端可以通过 `EventSource` 接收：

```ts
const eventSource = new EventSource(url);

eventSource.onmessage = ({ data }) => {
  output.textContent += data;
};
```

## Express 和 Nest 的区别

Express 更轻，主要提供路由和中间件能力。小项目可以直接写：

```ts
app.get('/ai/chat', async (req, res) => {
  const service = new AiService();
  const answer = await service.runChain(req.query.query);
  res.json({ answer });
});
```

但项目复杂后，会出现很多 controller、service、repository、config、logger、cache 等依赖。如果所有依赖都手动 `new`，代码会越来越难维护。

Nest 提供了更完整的组织方式：

```text
Controller 负责接请求
Service 负责业务逻辑
Module 负责组织模块边界
Provider 负责注册可注入依赖
IoC 容器负责创建和注入对象
```

所以可以理解为：

```text
Express：自己组织应用结构。
Nest：框架帮你建立企业级应用结构。
```

## 前端里也有 IoC 吗

有，只是前端里不一定叫 IoC。

React 里的 `props` 就有 IoC 思想：

```tsx
function UserCard({ user }: { user: { name: string } }) {
  return <div>{user.name}</div>;
}
```

`UserCard` 不自己创建 `user`，而是由父组件传入：

```tsx
<UserCard user={{ name: 'Tom' }} />
```

这就是把“数据从哪里来”的控制权交给外部。

React 里的 `Context Provider`、`children`、`render props`、回调函数、插件机制，也都有类似的控制反转思想。

## 易错点

1. `ConfigModule.forRoot()` 才是加载 `.env` 的地方，`controllers` 和 `providers` 只是 Nest 模块注册项。

2. `ChatOpenAI` 不等于所有模型都能用。它适合 OpenAI 或 OpenAI-compatible 接口。

3. 如果构造函数里马上要用某个依赖，应该优先用构造器注入，避免属性注入时依赖还没准备好。

4. `Observable` 在 SSE 场景下表示持续推送的数据流，不是一个普通的一次性返回值。

5. `PromptTemplate -> Model -> OutputParser` 是 LangChain LCEL 中很常见的三段式结构。

6. `useFactory` 适合创建依赖时需要读取配置或执行初始化逻辑的场景。

7. Prettier 和 ESLint 职责不同。Prettier 负责格式化，ESLint 更偏向代码质量和潜在错误检查，不建议把大量格式化问题都变成 ESLint error。

## 复习问题

1. `AppModule`、`AiModule`、`AiController`、`AiService` 分别负责什么？

2. `ConfigModule.forRoot({ isGlobal: true })` 的作用是什么？

3. 为什么 `AiService` 中更适合用构造器注入 `CHAT_MODEL`？

4. `CHAT_MODEL` provider 的 `provide`、`useFactory`、`inject` 分别是什么意思？

5. `PromptTemplate.pipe(model).pipe(new StringOutputParser())` 每一段分别做什么？

6. `Observable<{ data: string }>` 在 SSE 接口中代表什么？

7. 为什么说 Nest 的 IoC 不只是“少写一个 new”？

8. `ChatOpenAI` 和 `ChatAnthropic` 的使用场景有什么区别？
