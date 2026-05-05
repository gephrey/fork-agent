# 051. 二叉树的前序遍历

## 题目

返回二叉树前序遍历结果。

## 最优思路

前序顺序是根、左、右。递归最直观；若面试要求迭代，用栈先压右再压左。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(h)。

## JavaScript 题解

```js
function preorderTraversal(root) {
  const ans = [];
  const dfs = node => {
    if (!node) return;
    ans.push(node.val);
    dfs(node.left);
    dfs(node.right);
  };
  dfs(root);
  return ans;
}
```
