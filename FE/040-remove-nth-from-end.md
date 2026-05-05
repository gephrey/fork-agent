# 040. 删除链表倒数第 N 个节点

## 题目

删除链表倒数第 n 个节点并返回头节点。

## 最优思路

dummy 加快慢指针。fast 先走 n 步，然后 fast、slow 同步走到 fast.next 为空，slow.next 就是目标节点。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(1)。

## JavaScript 题解

```js
function removeNthFromEnd(head, n) {
  const dummy = {
    next: head
  };
  let fast = dummy, slow = dummy;
  for (let i = 0; i < n; i++) fast = fast.next;
  while (fast.next) {
    fast = fast.next;
    slow = slow.next;
  }
  slow.next = slow.next.next;
  return dummy.next;
}
```
