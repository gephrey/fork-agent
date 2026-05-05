# 070. 二叉搜索树中第 K 小的元素

## 题目

返回 BST 中第 k 小的值。

## 最优思路

BST 中序遍历是升序。迭代中序遍历，每弹出一个节点 k--，k 为 0 时返回。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(h+k)，空间 O(h)。

## JavaScript 题解

```js
function kthSmallest(root, k) {
  const st = [];
  while (root || st.length) {
    while (root) {
      st.push(root);
      root = root.left;
    }
    root = st.pop();
    if (--k === 0) return root.val;
    root = root.right;
  }
}
```
