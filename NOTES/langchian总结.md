# 为什么要用langchain
1. 它可以用统一的 ChatModel api 来调用各种大模型，屏蔽了底层差异。基于 LangChain 可以做到切换各种大模型，代码不变。
2. 对输入(PromptTemplate)、输出(OutputParser)做控制
在大模型的输出控制方面，model.withStructuredOutput 加上 OutputParser 就够用了。
3.  MCP，也就是可跨进程调用的 tool 可以直接复用,如果 MCP Server 跑在本地进程，就是用 stdio 进程通信，否则就是 http 通信,比如上面高德 MCP 是用了 http 通信，而 Chrome Devtools 的 MCP 用了 stdio 本地进程通信,代码里是用 @langchain/mcp-adapters 这个包来和 MCP Server 通信
4.ChatMessageHistory
-InMemoryChatHistoryHistory
-RedisChatHistoryHistory 
-FileSystemChatHistoryHistory
-TypeORMChatHistoryHistory
它可以把 messages 存到内存、redis、文件、数据库等。
5. memory 的管理策略也有三种：
截断，去掉之前的一些 message
总结，调用大模型对之前的 messages 生成摘要
检索，基于向量数据库根据 query 检索之前聊的内容来继续聊长时记忆基本都是要用向量数据库检索的。
6.  RAG/Milvus
把一段内容向量化，在坐标空间内就可以通过夹角来判断相似度：余弦相似度。
-各种来源的内容，通过 loader 加载，用 Splitter 分割后再用嵌入模型向量化，存到 Milvus 之类的向量数据库。
-根据 query 向量化之后去做余弦相似度匹配，就可以检索出相关文档，让大模型生成回答

我们是直接用的 @zilliz/milvus2-sdk-node 这个 Milvus 的包实际上 LangChain 有一层封装，在 @langchin/comunity 包下：

7.LCEL
调用方式有 invoke（同步调用）、stream（流式）、batch（批量调用） 三种：