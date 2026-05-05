# 053. 二叉树的后序遍历

## 题目

返回二叉树后序遍历结果。

## 最优思路

后序顺序是左、右、根。递归先处理左右子树，再访问当前节点。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(h)。

## JavaScript 题解

```js
function postorderTraversal(root) {
  const ans = [];
  const dfs = node => {
    if (!node) return;
    dfs(node.left);
    dfs(node.right);
    ans.push(node.val);
  };
  dfs(root);
  return ans;
}
```
