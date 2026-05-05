# 042. K 个一组翻转链表

## 题目

每 k 个节点一组翻转，不足 k 个保持原状。

## 最优思路

先检查剩余是否足够 k 个。足够则局部反转，再把前后链表接起来。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(1)。

## JavaScript 题解

```js
function reverseKGroup(head, k) {
  const dummy = {
    next: head
  };
  let pre = dummy;
  while (true) {
    let tail = pre;
    for (let i = 0; i < k && tail; i++) tail = tail.next;
    if (!tail) break;
    const next = tail.next, start = pre.next;
    let prev = next, cur = start;
    while (cur !== next) {
      const t = cur.next;
      cur.next = prev;
      prev = cur;
      cur = t;
    }
    pre.next = tail;
    pre = start;
  }
  return dummy.next;
}
```
