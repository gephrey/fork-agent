# 048. 重排链表

## 题目

将 L0->L1->...->Ln 重排为 L0->Ln->L1->Ln-1...。

## 最优思路

三步：找中点，反转后半链表，交替合并前半和反转后的后半。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(1)。

## JavaScript 题解

```js
function reorderList(head) {
  if (!head || !head.next) return;
  let slow = head, fast = head;
  while (fast.next && fast.next.next) {
    slow = slow.next;
    fast = fast.next.next;
  }
  let cur = slow.next;
  slow.next = null;
  let prev = null;
  while (cur) {
    const next = cur.next;
    cur.next = prev;
    prev = cur;
    cur = next;
  }
  let a = head, b = prev;
  while (b) {
    const an = a.next, bn = b.next;
    a.next = b;
    b.next = an;
    a = an;
    b = bn;
  }
}
```
