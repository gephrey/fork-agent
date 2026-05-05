# 069. 验证二叉搜索树

## 题目

判断二叉树是否为合法 BST。

## 最优思路

递归传递上下界。左子树所有值必须在 (low,node.val)，右子树在 (node.val,high)。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(h)。

## JavaScript 题解

```js
function isValidBST(root) {
  const dfs = (node, low, high) => {
    if (!node) return true;
    if (node.val <= low || node.val >= high) return false;
    return dfs(node.left, low, node.val) && dfs(node.right, node.val, high);
  };
  return dfs(root, -Infinity, Infinity);
}
```
