# 071. 二叉搜索树的中序后继

## 题目

给定 BST 和节点 p，找中序遍历中 p 的下一个节点。

## 最优思路

若当前值大于 p.val，当前可能是后继，记录并去左边找更小候选；否则去右边。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(h)，空间 O(1)。

## JavaScript 题解

```js
function inorderSuccessor(root, p) {
  let ans = null;
  while (root) {
    if (root.val > p.val) {
      ans = root;
      root = root.left;
    } else root = root.right;
  }
  return ans;
}
```
