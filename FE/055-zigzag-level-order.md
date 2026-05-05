# 055. 二叉树的锯齿形层序遍历

## 题目

按层遍历二叉树，方向左右交替。

## 最优思路

BFS 分层。偶数层尾插，奇数层头插，或者正常收集后按层反转。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(n)。

## JavaScript 题解

```js
function zigzagLevelOrder(root) {
  if (!root) return [];
  const q = [root], ans = [];
  for (let head = 0, levelNo = 0; head < q.length; levelNo++) {
    const size = q.length - head, level = [];
    for (let i = 0; i < size; i++) {
      const node = q[head++];
      if (levelNo % 2) level.unshift(node.val);
else level.push(node.val);
      if (node.left) q.push(node.left);
      if (node.right) q.push(node.right);
    }
    ans.push(level);
  }
  return ans;
}
```
