import { Controller, Get, Query, Sse } from '@nestjs/common';
import { from, map, Observable } from 'rxjs';
import { AiService } from './ai.service';

@Controller('ai')
export class AiController {
  constructor(private readonly aiService: AiService) {}

  @Get('chat')
  async chat(@Query('query') query: string) {
    const answer = await this.aiService.runChain(query);
    return answer;
  }

  // 这里的 Observable<{ data: string }> 代表：这个接口返回的不是一次性结果，而是一段会持续推送的数据流。
  @Sse('chat/stream')
  chatStream(@Query('query') query: string): Observable<{ data: string }> {
    return from(this.aiService.streamChain(query)).pipe(map((chunk) => ({ data: chunk })));
  }
}
