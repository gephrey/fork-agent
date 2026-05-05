# 101. 异步 Pipeline 执行器

## 题目

实现一个异步 Pipeline / RunnableSequence 执行器。它接收一组步骤，调用 `invoke(input)` 时按顺序执行每一步，并把上一步的输出作为下一步的输入。

每个步骤可以是：

- 普通函数：`value => nextValue`
- 异步函数：`async value => nextValue`
- 带 `invoke` 方法的对象：`{ invoke(value) { ... } }`

要求实现 `createPipeline(steps)`，返回的对象至少包含：

- `invoke(input)`：返回 Promise，resolve 为最后一步的输出。
- `pipe(next)`：返回一个新的 Pipeline，把 `next` 追加到当前步骤后面。

某一步抛错或返回 rejected Promise 时，后续步骤不再执行，并把错误继续抛出。

## 示例

```js
const trim = text => text.trim();
const upper = async text => text.toUpperCase();
const wrap = {
  invoke(text) {
    return `[${text}]`;
  }
};

const wrap2 = {
  invoke(text) {
    return `wrap2: text`;
  }
}

const pipeline = createPipeline([trim, upper]).pipe(wrap);

await pipeline.invoke('  hello  '); // '[HELLO]'
```

错误停止示例：

```js
const calls = [];
const pipeline = createPipeline([
  value => {
    calls.push('a');
    return value + 1;
  },
  () => {
    calls.push('b');
    throw new Error('boom');
  },
  value => {
    calls.push('c');
    return value * 2;
  }
]);

await pipeline.invoke(1); // 抛出 Error('boom')
calls; // ['a', 'b']
```

## 输入输出

输入：

- `steps`：步骤数组，每一项是函数或带 `invoke` 方法的对象。
- `input`：任意类型，作为第一个步骤的输入。

输出：

- `invoke(input)` 返回一个 Promise。
- Promise 成功时的值是最后一个步骤的返回值。
- Promise 失败时的原因是第一个失败步骤抛出的错误。

## 边界情况

- `steps` 为空时，`invoke(input)` 应直接返回原始 `input`。
- 步骤可以返回 Promise，也可以返回普通值。
- 带 `invoke` 方法的对象中，`invoke` 也可能是异步函数。
- `pipe(next)` 不应修改原 Pipeline，避免已有链路被意外改变。
- 遇到非法步骤时可以抛出 `TypeError`，例如既不是函数，也没有可调用的 `invoke`。

## 进阶要求

- 支持 `pipe` 传入另一个 Pipeline，并把它当成一个可 `invoke` 的步骤。
- 支持在错误对象中附加失败步骤下标，便于排查问题。
- 支持可选的执行上下文，例如给每个对象步骤的 `invoke` 传入 `this`。

## 最优思路

把所有步骤统一成“可调用一步”的形式：如果步骤是函数就直接调用，如果有 `invoke` 方法就调用 `step.invoke(input)`。主流程用 `for...of` 顺序 `await` 每一步，保证同步返回值、异步 Promise 和对象步骤都被统一处理。错误不需要吞掉，直接让异常冒泡即可，循环会自然停止。

`pipe(next)` 返回基于旧步骤数组和新步骤的新对象，不原地修改旧数组，这样链式组合更可预测。

## 关键点

- 每一步都必须等待完成后，才能把结果传给下一步。
- 同步函数和异步函数可以通过 `await` 统一处理。
- 对象步骤要判断 `invoke` 是否可调用，而不是记忆某个框架 API。
- 报错后不能继续执行后续步骤。
- `pipe(next)` 推荐返回新 Pipeline，避免共享可变状态。

## 复杂度

设步骤数为 n，不考虑每个步骤自身耗时，时间 O(n)，空间 O(n)，其中空间主要来自保存步骤数组。执行过程额外空间 O(1)。

## JavaScript 题解

```js
function createPipeline(steps = []) {
  const list = steps.slice();

  const runStep = async (step, input, index) => {
    try {
      if (typeof step === 'function') {
        return await step(input);
      }
      if (step && typeof step.invoke === 'function') {
        return await step.invoke(input);
      }
      throw new TypeError(`Invalid pipeline step at index ${index}`);
    } catch (error) {
      if (error && typeof error === 'object' && !('stepIndex' in error)) {
        error.stepIndex = index;
      }
      throw error;
    }
  };

  return {
    // 启动函数
    async invoke(input) {
      let current = input;
      for (let i = 0; i < list.length; i++) {
        current = await runStep(list[i], current, i);
      }
      return current;
    },

    pipe(next) {
      return createPipeline([...list, next]);
    }
  };
}
```

## 进阶版：完整 Runnable 协议

在进阶版中，所有步骤都实现统一的 Runnable 协议：

- `invoke(input)`：处理单个输入，返回单个输出。
- `stream(input)`：处理单个输入，返回异步迭代器，逐步产出最终输出片段。
- `batch(inputs)`：处理多个输入，返回与输入顺序一致的输出数组。

要求实现 `createRunnablePipeline(steps)`，返回的 Pipeline 也必须同时支持 `invoke`、`stream`、`batch` 和 `pipe`。

### 行为约定

- `invoke`：从左到右顺序执行，每一步的输出传给下一步。
- `batch`：按步骤批处理。先把整批输入交给第 1 步的 `batch`，再把结果数组交给第 2 步的 `batch`，直到最后一步。
- `stream`：如果没有额外的 `transform` 流式转换协议，前置步骤先用 `invoke` 得到完整中间结果，最后一步用 `stream` 产出片段。
- 空 Pipeline：
  - `invoke(input)` 返回 `input`。
  - `batch(inputs)` 返回原数组的浅拷贝。
  - `stream(input)` 只产出一次 `input`。
- 任意步骤报错时立即停止，并抛出包含失败步骤下标的错误。
- `pipe(next)` 返回新 Pipeline，不修改原 Pipeline。

### 最易懂示例

先把 Pipeline 想成一条流水线。输入从左边进去，每一步只做一件事：

```txt
输入 3
  -> 第 1 步：加 1，得到 4
  -> 第 2 步：乘 2，得到 8
  -> 第 3 步：转成字符串，得到 "结果是 8"
最终输出 "结果是 8"
```

对应代码：

下面示例假设已经使用后文的 `fromFunction` 和 `createRunnablePipeline` 实现。

```ts
const addOne = fromFunction((n: number) => n + 1);
const double = fromFunction(async (n: number) => n * 2);
const toText = fromFunction((n: number) => `结果是 ${n}`);

const pipeline = createRunnablePipeline([addOne, double, toText]);

await pipeline.invoke(3); // '结果是 8'
```

`batch` 就是一次放多个输入进去，但每个输入仍然走同一条流水线：

```txt
输入 [1, 2, 3]

第 1 步 addOne:
[1, 2, 3] -> [2, 3, 4]

第 2 步 double:
[2, 3, 4] -> [4, 6, 8]

第 3 步 toText:
[4, 6, 8] -> ["结果是 4", "结果是 6", "结果是 8"]
```

对应代码：

```ts
await pipeline.batch([1, 2, 3]);
// ['结果是 4', '结果是 6', '结果是 8']
```

`stream` 可以理解为最后一步不是一次性返回完整结果，而是一段一段吐出来：

```ts
const spell: Runnable<string, string, string> = {
  async invoke(text) {
    return text;
  },

  async *stream(text) {
    for (const char of text) {
      yield char;
    }
  },

  async batch(inputs) {
    return [...inputs];
  }
};

const streamPipeline = createRunnablePipeline([
  fromFunction((name: string) => name.trim()),
  fromFunction((name: string) => `Hi, ${name}`),
  spell
]);

const chunks = [];
for await (const chunk of streamPipeline.stream('  Tom  ')) {
  chunks.push(chunk);
}

chunks; // ['H', 'i', ',', ' ', 'T', 'o', 'm']
chunks.join(''); // 'Hi, Tom'
```

这三个方法的区别可以简单记成：

- `invoke`：一个输入，得到一个最终结果。
- `batch`：一批输入，得到一批最终结果。
- `stream`：一个输入，逐段得到最终结果。

### TypeScript 题解

```ts
type Awaitable<T> = T | PromiseLike<T>;

export interface Runnable<I, O, Chunk = O> {
  invoke(input: I): Awaitable<O>;
  stream(input: I): AsyncIterable<Chunk>;
  batch(inputs: readonly I[]): Awaitable<O[]>;
}

export interface RunnablePipeline<I, O, Chunk = O>
  extends Runnable<I, O, Chunk> {
  readonly length: number;
  pipe<Next, NextChunk = Next>(
    next: Runnable<O, Next, NextChunk>
  ): RunnablePipeline<I, Next, NextChunk>;
}

type AnyRunnable = Runnable<any, any, any>;
type RunnableInput<T> = T extends Runnable<infer I, any, any> ? I : never;
type RunnableOutput<T> = T extends Runnable<any, infer O, any> ? O : never;
type RunnableChunk<T> = T extends Runnable<any, any, infer C> ? C : never;
type Last<T extends readonly unknown[]> = T extends readonly [...unknown[], infer L]
  ? L
  : never;

export class RunnablePipelineError extends Error {
  readonly stepIndex: number;
  readonly method: 'invoke' | 'stream' | 'batch';
  readonly step: unknown;
  readonly cause: unknown;

  constructor(
    method: 'invoke' | 'stream' | 'batch',
    stepIndex: number,
    step: unknown,
    cause: unknown
  ) {
    const message = cause instanceof Error && cause.message ? `: ${cause.message}` : '';
    super(`Runnable pipeline ${method} failed at step ${stepIndex}${message}`);
    this.name = 'RunnablePipelineError';
    this.method = method;
    this.stepIndex = stepIndex;
    this.step = step;
    this.cause = cause;
  }
}

function wrapPipelineError(
  method: 'invoke' | 'stream' | 'batch',
  stepIndex: number,
  step: unknown,
  error: unknown
): never {
  if (error instanceof RunnablePipelineError) {
    throw error;
  }
  throw new RunnablePipelineError(method, stepIndex, step, error);
}

async function* once<T>(value: T): AsyncIterable<T> {
  yield value;
}

export function fromFunction<I, O>(
  fn: (input: I) => Awaitable<O>
): Runnable<I, O, O> {
  return Object.freeze({
    async invoke(input: I): Promise<O> {
      return await fn(input);
    },

    async *stream(input: I): AsyncIterable<O> {
      yield await fn(input);
    },

    async batch(inputs: readonly I[]): Promise<O[]> {
      return await Promise.all(inputs.map(input => fn(input)));
    }
  });
}

export function createRunnablePipeline<I = unknown>(): RunnablePipeline<I, I, I>;
export function createRunnablePipeline<
  Steps extends readonly [AnyRunnable, ...AnyRunnable[]]
>(
  steps: Steps
): RunnablePipeline<
  RunnableInput<Steps[0]>,
  RunnableOutput<Last<Steps>>,
  RunnableChunk<Last<Steps>>
>;
export function createRunnablePipeline<I, O, Chunk = O>(
  steps: readonly AnyRunnable[]
): RunnablePipeline<I, O, Chunk>;
export function createRunnablePipeline<I, O = I, Chunk = O>(
  steps: readonly AnyRunnable[] = []
): RunnablePipeline<I, O, Chunk> {
  const list = Object.freeze([...steps]);

  const pipeline: RunnablePipeline<I, O, Chunk> = Object.freeze({
    get length() {
      return list.length;
    },

    async invoke(input: I): Promise<O> {
      let current: unknown = input;

      for (let index = 0; index < list.length; index++) {
        const step = list[index];
        try {
          current = await step.invoke(current);
        } catch (error) {
          wrapPipelineError('invoke', index, step, error);
        }
      }

      return current as O;
    },

    async *stream(input: I): AsyncIterable<Chunk> {
      if (list.length === 0) {
        yield input as unknown as Chunk;
        return;
      }

      let current: unknown = input;
      const lastIndex = list.length - 1;

      for (let index = 0; index < lastIndex; index++) {
        const step = list[index];
        try {
          current = await step.invoke(current);
        } catch (error) {
          wrapPipelineError('stream', index, step, error);
        }
      }

      const lastStep = list[lastIndex];
      try {
        for await (const chunk of lastStep.stream(current)) {
          yield chunk as Chunk;
        }
      } catch (error) {
        wrapPipelineError('stream', lastIndex, lastStep, error);
      }
    },

    async batch(inputs: readonly I[]): Promise<O[]> {
      let current: readonly unknown[] = [...inputs];

      for (let index = 0; index < list.length; index++) {
        const step = list[index];
        try {
          current = await step.batch(current);
        } catch (error) {
          wrapPipelineError('batch', index, step, error);
        }
      }

      return [...current] as O[];
    },

    pipe<Next, NextChunk = Next>(
      next: Runnable<O, Next, NextChunk>
    ): RunnablePipeline<I, Next, NextChunk> {
      return createRunnablePipeline<I, Next, NextChunk>([...list, next]);
    }
  });

  return pipeline;
}
```

使用示例：

```ts
const trim = fromFunction((text: string) => text.trim());
const upper = fromFunction(async (text: string) => text.toUpperCase());

const chars: Runnable<string, string, string> = {
  async invoke(text) {
    return `[${text}]`;
  },

  async *stream(text) {
    yield '[';
    for (const char of text) {
      yield char;
    }
    yield ']';
  },

  async batch(inputs) {
    return inputs.map(text => `[${text}]`);
  }
};

const pipeline = createRunnablePipeline([trim, upper]).pipe(chars);

await pipeline.invoke('  hello  '); // '[HELLO]'
await pipeline.batch([' a ', ' b ']); // ['[A]', '[B]']

const chunks = [];
for await (const chunk of pipeline.stream('  hi  ')) {
  chunks.push(chunk);
}
chunks.join(''); // '[HI]'
```
