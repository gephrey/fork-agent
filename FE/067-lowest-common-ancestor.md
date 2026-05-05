# 067. 二叉树的最近公共祖先

## 题目

给定二叉树两个节点 p、q，返回最近公共祖先。

## 最优思路

递归。若当前节点为空或等于 p/q，返回当前节点；左右子树都有返回则当前是 LCA，否则返回非空一侧。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(h)。

## JavaScript 题解

```js
function lowestCommonAncestor(root, p, q) {
  if (!root || root === p || root === q) return root;
  const l = lowestCommonAncestor(root.left, p, q);
  const r = lowestCommonAncestor(root.right, p, q);
  return l && r ? root: l || r;
}
```
