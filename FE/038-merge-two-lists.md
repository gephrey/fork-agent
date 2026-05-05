# 038. 合并两个有序链表

## 题目

合并两个升序链表并返回新链表。

## 最优思路

dummy 尾插。每次比较两个链表头，取较小节点接到结果尾部。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(m+n)，空间 O(1)。

## JavaScript 题解

```js
function mergeTwoLists(a, b) {
  const dummy = {};
  let cur = dummy;
  while (a && b) {
    if (a.val <= b.val) {
      cur.next = a;
      a = a.next;
    } else {
      cur.next = b;
      b = b.next;
    }
    cur = cur.next;
  }
  cur.next = a || b;
  return dummy.next;
}
```
