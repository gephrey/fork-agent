# 039. 合并 K 个升序链表

## 题目

合并 k 个升序链表。

## 最优思路

分治合并。每轮两两合并链表，直到只剩一个，避免简单顺序合并退化。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(N log k)，空间 O(1) 迭代版。

## JavaScript 题解

```js
function mergeKLists(lists) {
  if (!lists.length) return null;
  const merge = (a, b) => {
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
  };
  while (lists.length > 1) {
    const next = [];
    for (let i = 0; i < lists.length; i += 2) next.push(merge(lists[i], lists[i + 1] || null));
    lists = next;
  }
  return lists[0];
}
```
