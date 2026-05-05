# 037. 反转链表 II

## 题目

反转链表从 left 到 right 的部分。

## 最优思路

使用 dummy。先走到反转区间前一个节点 pre，再用头插法把区间内节点逐个插到 pre 后面。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(1)。

## JavaScript 题解

```js
function reverseBetween(head, left, right) {
  const dummy = {
    next: head
  };
  let pre = dummy;
  for (let i = 1; i < left; i++) pre = pre.next;
  let cur = pre.next;
  for (let i = 0; i < right - left; i++) {
    const move = cur.next;
    cur.next = move.next;
    move.next = pre.next;
    pre.next = move;
  }
  return dummy.next;
}
```
