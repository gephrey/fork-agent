import { Inject, Injectable } from '@nestjs/common';
import { ChatOpenAI } from '@langchain/openai';
import { PromptTemplate } from '@langchain/core/prompts';
import type { Runnable } from '@langchain/core/runnables';
import { StringOutputParser } from '@langchain/core/output_parsers';

@Injectable()
export class AiService {
  private readonly chain: Runnable;

  /*
   因为 this.chain 是在构造函数里创建的，而创建 chain
  时必须马上拿到 model。

  所以这里适合用“构造器注入”：

  constructor(@Inject('CHAT_MODEL') model:
  ChatOpenAI) {}

  意思是：Nest 创建 AiService 对象时，先把
  CHAT_MODEL 准备好，然后作为参数传进构造函数。

  如果改成属性注入，比如：

  @Inject('CHAT_MODEL')
  private readonly model: ChatOpenAI;

  constructor() {
    this.chain = prompt.pipe(this.model).pipe(new
  StringOutputParser());
  }

  这里就会有问题。

  原因是：执行 constructor() 的时候，AiService 对象
  还在创建过程中，属性注入的 this.model 还没有被
  Nest 填进去，所以此时 this.model 可能是
  undefined。

  */
  constructor(
    // @Inject(ConfigService) configService: ConfigService,
    @Inject('CHAT_MODEL') model: ChatOpenAI,
  ) {
    const prompt = PromptTemplate.fromTemplate('请回答以下问题：\n\n{query}');
    // const model = new ChatOpenAI({
    //   temperature: 0.7,
    //   model: configService.get('MODEL_NAME'),
    //   apiKey: configService.get('OPENAI_API_KEY'),
    //   configuration: {
    //     baseURL: configService.get('OPENAI_BASE_URL')
    //   },
    // });
    this.chain = prompt.pipe(model).pipe(new StringOutputParser());
  }

  async runChain(query: string): Promise<string> {
    return this.chain.invoke({ query });
  }

  async *streamChain(query: string): AsyncGenerator<string> {
    const stream = await this.chain.stream({ query });
    for await (const chunk of stream) {
      yield chunk;
    }
  }
}
