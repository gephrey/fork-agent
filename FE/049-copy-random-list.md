# 049. 复制带随机指针的链表

## 题目

复制每个节点包含 next 和 random 指针的链表。

## 最优思路

哈希表建立旧节点到新节点映射。第一遍创建节点，第二遍连接 next 和 random。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(n)。

## JavaScript 题解

```js
function copyRandomList(head) {
  if (!head) return null;
  const map = new Map();
  for (let p = head; p; p = p.next) map.set(p, {
    val: p.val, next: null, random: null
  });
  for (let p = head; p; p = p.next) {
    map.get(p).next = map.get(p.next) || null;
    map.get(p).random = map.get(p.random) || null;
  }
  return map.get(head);
}
```
