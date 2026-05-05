常规结构

- foo
  - dto
    - foo-action.dto.ts
      定义比如创建、更新时的数据结构
      imports from class-validator
  - entities
    - foothing.entity.ts
      imports from typeorm
  - foo.service.ts
    - u can see @Injectable()
    - u may see  @Inject(foothing)

- u may see constructor( @Inject(foothing) foothing :foothing)
- foo.module.ts
  - @Module({})
  - imports:[] 别个 module
  - controllers: []自己的 controllers
  - providers:[] 一堆 services

