// ============================================
// 导入外部依赖模块
// ============================================

// 加载 .env 环境变量文件，使环境变量可以通过 process.env 访问
// 这允许我们从 .env 文件中读取敏感的 API 密钥等配置
import "dotenv/config";

// 从 @zilliz/milvus2-sdk-node 包中导入 Milvus 客户端及相关常量
// MilvusClient: 用于连接和操作 Milvus 向量数据库的主类
// DataType: 定义 Milvus 支持的数据类型枚举（如 VarChar, FloatVector, Array 等）
// MetricType: 定义向量距离度量类型枚举（如 COSINE, L2, IP 等）
// IndexType: 定义索引类型枚举（如 IVF_FLAT, HNSW, IVF_SQ8 等）
import { MilvusClient, DataType, MetricType, IndexType } from '@zilliz/milvus2-sdk-node';

// 从 LangChain 的 OpenAI 包中导入嵌入模型类
// 用于将文本转换为高维向量表示（embeddings），以便进行向量相似度搜索
import { OpenAIEmbeddings } from "@langchain/openai";

// ============================================
// 定义常量
// ============================================

// Milvus 集合的名称，用于标识和操作数据库中的特定集合
// 集合类似于关系型数据库中的表，用于存储日记数据
const COLLECTION_NAME = 'ai_diary';

// 向量的维度，表示每个文本嵌入向量的长度
// 1024 维向量是比较常见的选择，兼顾精度和性能
const VECTOR_DIM = 1024;

// ============================================
// 初始化 OpenAI 嵌入模型
// ============================================

// 创建 OpenAI 嵌入模型实例
// 这个实例用于将文本内容转换为向量表示
const embeddings = new OpenAIEmbeddings({
  // OpenAI API 密钥，从环境变量中读取
  // 用于认证 OpenAI API 请求，确保有权限使用服务
  apiKey: process.env.OPENAI_API_KEY,

  // 指定使用的嵌入模型名称，从环境变量中读取
  // 不同的模型可能有不同的维度和性能特点
  model: process.env.EMBEDDINGS_MODEL_NAME,

  // OpenAI API 的配置选项
  configuration: {
    // 自定义 API 基础 URL，允许使用代理或第三方兼容 API
    // 如果使用官方 OpenAI API，通常不需要设置此项
    baseURL: process.env.OPENAI_BASE_URL
  },

  // 指定嵌入向量的维度，必须与 VECTOR_DIM 常量一致
  // 这确保生成的向量维度与 Milvus 集合定义匹配
  dimensions: VECTOR_DIM
});

// ============================================
// 初始化 Milvus 客户端
// ============================================

// 创建 Milvus 客户端实例
// Milvus 是一个开源的向量数据库，专门用于存储和搜索高维向量
const milvusClient = new MilvusClient({
  // Milvus 服务的地址和端口
  // localhost: 表示连接本地运行的 Milvus 服务
  // 19530: Milvus 默认的 gRPC 端口号
  address: 'localhost:19530'
});

// ============================================
// 定义辅助函数
// ============================================

// 异步函数：获取文本的嵌入向量表示
// @param text - 需要转换的文本内容
// @returns - 返回文本对应的向量数组
async function getEmbedding(text) {
  // 调用嵌入模型的 embedQuery 方法
  // 这个方法接收一段文本，返回该文本的向量表示
  // 返回结果是一个浮点数数组，长度为 VECTOR_DIM (1024)
  const result = await embeddings.embedQuery(text);

  // 将结果返回给调用者
  return result;
}

// ============================================
// 主函数：执行数据库操作流程
// ============================================

async function main() {
  // 使用 try-catch 块捕获可能发生的错误
  // 确保程序不会因为未处理的异常而崩溃
  try {
    // 输出连接提示信息，让用户知道程序正在尝试连接
    console.log('Connecting to Milvus...');

    // 连接到 Milvus 服务器
    // client.connectPromise 是一个 Promise 对象
    // await 会等待连接成功或失败
    await milvusClient.connectPromise;

    // 连接成功后输出成功提示
    // \n 表示换行，使输出更美观
    console.log('✓ Connected\n');

    // ----------------------------------------
    // 步骤 1: 创建集合（Collection）
    // ----------------------------------------

    // 输出集合创建提示信息
    console.log('Creating collection...');

    // 调用 createCollection 方法创建新的集合
    // 集合是 Milvus 中存储数据的基本单元
    await milvusClient.createCollection({
      // 指定要创建的集合名称
      collection_name: COLLECTION_NAME,

      // 定义集合的字段结构，类似于关系型数据库的表结构
      fields: [
        // 字段 1: id - 日记的唯一标识符
        // VarChar 类型用于存储字符串，is_primary_key 表示这是主键
        // 主键用于唯一标识每条记录，且必须唯一
        { name: 'id', data_type: DataType.VarChar, max_length: 50, is_primary_key: true },

        // 字段 2: vector - 日记内容的向量表示
        // FloatVector 类型专门用于存储浮点数向量
        // dim 参数指定向量的维度，必须与 VECTOR_DIM 一致
        { name: 'vector', data_type: DataType.FloatVector, dim: VECTOR_DIM },

        // 字段 3: content - 日记的文本内容
        // VarChar 类型存储最多 5000 个字符的文本
        { name: 'content', data_type: DataType.VarChar, max_length: 5000 },

        // 字段 4: date - 日记的日期
        // VarChar 类型存储日期字符串，最多 50 个字符
        { name: 'date', data_type: DataType.VarChar, max_length: 50 },

        // 字段 5: mood - 日记记录的心情状态
        // 存储心情标签，如 "happy", "sad", "excited" 等
        { name: 'mood', data_type: DataType.VarChar, max_length: 50 },

        // 字段 6: tags - 日记的标签数组
        // Array 类型用于存储数组数据
        // element_type 指定数组元素的类型（VarChar 字符串）
        // max_capacity 指定数组最多能容纳的元素个数（10个）
        // max_length 指定每个元素的 最大字符长度（50个）
        { name: 'tags', data_type: DataType.Array, element_type: DataType.VarChar, max_capacity: 10, max_length: 50 }
      ]
    });

    // 输出集合创建成功的提示
    console.log('Collection created');

    // ----------------------------------------
    // 步骤 2: 创建索引（Index）
    // ----------------------------------------

    // 输出索引创建提示，换行使输出更清晰
    console.log('\nCreating index...');

    // 创建向量字段的索引
    // 索引可以大幅提高向量相似度搜索的速度
    await milvusClient.createIndex({
      // 指定要创建索引的集合名称
      collection_name: COLLECTION_NAME,

      // 指定要创建索引的字段名称
      // 这里为 vector 字段创建索引，因为这是用于搜索的向量字段
      field_name: 'vector',

      // 指定索引类型
      // IVF_FLAT 是一种基于倒排索引的算法
      // 适用于中等规模的数据集，平衡了搜索精度和性能
      index_type: IndexType.IVF_FLAT,

      // 指定向量距离度量类型
      // COSINE（余弦相似度）衡量两个向量方向的相似性
      // 值范围通常在 -1 到 1 之间，1 表示完全相同
      metric_type: MetricType.COSINE,

      // 索引的特定参数
      // nlist: 聚类中心数量，影响索引的粒度和搜索性能
      // 更多的聚类中心可以提高搜索精度，但会增加内存占用
      params: { nlist: 1024 }
    });

    // 输出索引创建成功的提示
    console.log('Index created');

    // ----------------------------------------
    // 步骤 3: 加载集合（Load Collection）
    // ----------------------------------------

    // 输出集合加载提示
    console.log('\nLoading collection...');

    // 将集合加载到内存中，以便进行搜索操作
    // 在 Milvus 中，数据默认存储在磁盘上
    // 进行搜索前需要先加载到内存以提高性能
    await milvusClient.loadCollection({ collection_name: COLLECTION_NAME });

    // 输出集合加载成功的提示
    console.log('Collection loaded');

    // ----------------------------------------
    // 步骤 4: 准备并插入日记数据
    // ----------------------------------------

    // 输出数据插入提示
    console.log('\nInserting diary entries...');

    // 定义日记内容数组
    // 每个日记对象包含：id（唯一标识）、content（内容）、date（日期）、mood（心情）、tags（标签）
    const diaryContents = [
      // 日记条目 1: 关于散步和好心情
      {
        // 唯一标识符，格式为 diary_序号
        // 用于在数据库中唯一标识这条记录
        id: 'diary_001',

        // 日记的文本内容
        // 这是要被嵌入成向量的主要文本
        content: '今天天气很好，去公园散步了，心情愉快。看到了很多花开了，春天真美好。',

        // 日记的日期，格式为 YYYY-MM-DD
        date: '2026-01-10',

        // 记录时的心情状态
        mood: 'happy',

        // 标签数组，用于分类和检索
        // 可以有多个标签，用逗号分隔
        tags: ['生活', '散步']
      },

      // 日记条目 2: 关于工作成就
      {
        id: 'diary_002',

        // 工作相关的日记内容
        content:'今天打了五局王者，赢了两局，输了两局，心情很差。',

        date: '2026-01-11',

        // 使用英文标签以便后续处理
        mood: 'excited',

        tags: ['工作', '成就']
      },

      // 日记条目 3: 关于户外活动
      {
        id: 'diary_003',

        // 周末爬山的内容
        content: '周末和朋友去爬山，天气很好，心情也很放松。享受大自然的感觉真好。',

        date: '2026-01-12',

        mood: 'relaxed',

        tags: ['户外', '朋友']
      },

      // 日记条目 4: 关于技术学习
      {
        id: 'diary_004',

        // 学习 Milvus 数据库的内容
        content: '今天学习了 Milvus 向量数据库，感觉很有意思。向量搜索技术真的很强大。',

        date: '2026-01-12',

        mood: 'curious',

        tags: ['学习', '技术']
      },

      // 日记条目 5: 关于烹饪
      {
        id: 'diary_005',

        // 做晚餐的内容
        content: '晚上做了一顿丰盛的晚餐，尝试了新菜谱。家人都说很好吃，很有成就感。',

        date: '2026-01-13',

        mood: 'proud',

        tags: ['美食', '家庭']
      }
    ];

    // 输出嵌入生成提示
    console.log('Generating embeddings...');

    // 为每个日记内容生成向量嵌入
    // Promise.all 允许并行处理多个异步任务，提高效率
    // diaryContents.map 遍历每个日记条目
    const diaryData = await Promise.all(
      // 使用 map 对每个日记进行异步处理
      diaryContents.map(async (diary) => ({
        // 展开运算符，保留原始的日记属性
        // 包括 id, content, date, mood, tags
        ...diary,

        // 为日记内容生成向量表示
        // 这是向量数据库搜索的关键
        // getEmbedding 将文本转换为 VECTOR_DIM 维的向量
        vector: await getEmbedding(diary.content)
      }))
    );

    // 调用 Milvus 客户端的 insert 方法插入数据
    const insertResult = await milvusClient.insert({
      // 指定要插入数据的集合名称
      collection_name: COLLECTION_NAME,

      // 要插入的数据数组
      // 每个对象包含: id, content, date, mood, tags, vector
      data: diaryData
    });

    // 输出插入结果的摘要信息
    // insertResult.insert_cnt 表示成功插入的记录数量
    console.log(`✓ Inserted ${insertResult.insert_cnt} records\n`);

    // ----------------------------------------
    // 错误处理
    // ----------------------------------------

  } catch (error) {
    // 捕获并输出错误信息
    // error.message 提供人类可读的错误描述
    console.error('Error:', error.message);
  }
}

// ============================================
// 程序入口
// ============================================

// 调用 main 函数，启动程序执行
// 这是 JavaScript 程序的常规入口点
main();
