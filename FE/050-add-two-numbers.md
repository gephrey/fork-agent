# 050. 两数相加

## 题目

两个链表表示逆序非负整数，返回相加后的链表。

## 最优思路

同步遍历两个链表，逐位相加并维护进位 carry。每步创建一个新节点。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(max(m,n))，空间 O(max(m,n))。

## JavaScript 题解

```js
function addTwoNumbers(l1, l2) {
  const dummy = {};
  let cur = dummy, carry = 0;
  while (l1 || l2 || carry) {
    const sum = (l1 ? l1.val: 0) + (l2 ? l2.val: 0) + carry;
    cur.next = {
      val: sum % 10, next: null
    };
    carry = Math.floor(sum / 10);
    cur = cur.next;
    if (l1) l1 = l1.next;
    if (l2) l2 = l2.next;
  }
  return dummy.next;
}
```
