# 079. 找树左下角的值

## 题目

返回二叉树最后一层最左边的值。

## 最优思路

BFS 每层从左到右遍历，记录每层第一个节点；最后记录的就是答案。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(n)。

## JavaScript 题解

```js
function findBottomLeftValue(root) {
  const q = [root];
  let ans = root.val;
  for (let head = 0; head < q.length;) {
    const size = q.length - head;
    for (let i = 0; i < size; i++) {
      const node = q[head++];
      if (i === 0) ans = node.val;
      if (node.left) q.push(node.left);
      if (node.right) q.push(node.right);
    }
  }
  return ans;
}
```
