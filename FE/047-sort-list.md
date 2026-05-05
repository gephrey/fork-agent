# 047. 排序链表

## 题目

对链表进行升序排序。

## 最优思路

归并排序最适合链表。快慢指针找中点拆分，递归排序左右链表，再合并两个有序链表。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n log n)，空间 O(log n)。

## JavaScript 题解

```js
function sortList(head) {
  if (!head || !head.next) return head;
  let slow = head, fast = head.next;
  while (fast && fast.next) {
    slow = slow.next;
    fast = fast.next.next;
  }
  const mid = slow.next;
  slow.next = null;
  return merge(sortList(head), sortList(mid));
}
function merge(a, b) {
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
