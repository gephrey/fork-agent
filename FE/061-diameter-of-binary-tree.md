# 061. 二叉树的直径

## 题目

求任意两个节点之间最长路径的边数。

## 最优思路

DFS 返回当前节点高度，同时用 leftHeight + rightHeight 更新全局直径。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(h)。

## JavaScript 题解

```js
function diameterOfBinaryTree(root) {
  let ans = 0;
  const depth = node => {
    if (!node) return 0;
    const l = depth(node.left), r = depth(node.right);
    ans = Math.max(ans, l + r);
    return Math.max(l, r) + 1;
  };
  depth(root);
  return ans;
}
```
