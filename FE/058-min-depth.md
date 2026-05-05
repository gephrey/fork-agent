# 058. 二叉树的最小深度

## 题目

求根节点到最近叶子节点的节点数。

## 最优思路

BFS 最先遇到的叶子就是最小深度。DFS 要注意单子树节点不能把空子树深度当 0 取最小。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(n)。

## JavaScript 题解

```js
function minDepth(root) {
  if (!root) return 0;
  const q = [[root, 1]];
  for (let i = 0; i < q.length; i++) {
    const[node, d] = q[i];
    if (!node.left && !node.right) return d;
    if (node.left) q.push([node.left, d + 1]);
    if (node.right) q.push([node.right, d + 1]);
  }
}
```
