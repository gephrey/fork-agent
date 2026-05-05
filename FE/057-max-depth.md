# 057. 二叉树的最大深度

## 题目

求根节点到最远叶子节点的节点数。

## 最优思路

递归定义：空树深度 0，非空树深度为左右子树最大深度加 1。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(h)。

## JavaScript 题解

```js
function maxDepth(root) {
  return root ? Math.max(maxDepth(root.left), maxDepth(root.right)) + 1: 0;
}
```
