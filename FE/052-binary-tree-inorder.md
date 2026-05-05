# 052. 二叉树的中序遍历

## 题目

返回二叉树中序遍历结果。

## 最优思路

中序顺序是左、根、右。递归到最左，再访问根，再访问右子树。BST 的中序结果有序。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(h)。

## JavaScript 题解

```js
function inorderTraversal(root) {
  const ans = [];
  const dfs = node => {
    if (!node) return;
    dfs(node.left);
    ans.push(node.val);
    dfs(node.right);
  };
  dfs(root);
  return ans;
}
```
