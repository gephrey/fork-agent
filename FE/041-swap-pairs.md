# 041. 两两交换链表中的节点

## 题目

每两个相邻节点交换一次。

## 最优思路

dummy 指向头前，pre 每次指向待交换两节点前一个。调整 a、b、pre 三者指针后向后推进。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(1)。

## JavaScript 题解

```js
function swapPairs(head) {
  const dummy = {
    next: head
  };
  let pre = dummy;
  while (pre.next && pre.next.next) {
    const a = pre.next, b = a.next;
    a.next = b.next;
    b.next = a;
    pre.next = b;
    pre = a;
  }
  return dummy.next;
}
```
