# 062. 平衡二叉树

## 题目

判断每个节点左右子树高度差是否不超过 1。

## 最优思路

后序 DFS 返回高度；一旦子树不平衡返回 -1 作为哨兵，向上传播失败。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(h)。

## JavaScript 题解

```js
function isBalanced(root) {
  const height = node => {
    if (!node) return 0;
    const l = height(node.left);
    if (l < 0) return -1;
    const r = height(node.right);
    if (r < 0 || Math.abs(l - r) > 1) return -1;
    return Math.max(l, r) + 1;
  };
  return height(root) >= 0;
}
```
