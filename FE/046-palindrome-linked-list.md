# 046. 回文链表

## 题目

判断链表是否为回文。

## 最优思路

快慢指针找中点，反转后半段，再从头和后半段逐个比较。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(1)。

## JavaScript 题解

```js
function isPalindrome(head) {
  let slow = head, fast = head;
  while (fast && fast.next) {
    slow = slow.next;
    fast = fast.next.next;
  }
  let prev = null, cur = slow;
  while (cur) {
    const next = cur.next;
    cur.next = prev;
    prev = cur;
    cur = next;
  }
  while (prev) {
    if (head.val !== prev.val) return false;
    head = head.next;
    prev = prev.next;
  }
  return true;
}
```
