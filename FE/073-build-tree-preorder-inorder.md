# 073. 从前序与中序遍历构造二叉树

## 题目

给定 preorder 和 inorder，构造原二叉树。

## 最优思路

前序第一个是根。用哈希表快速定位根在中序中的位置，左侧长度决定前序的左右子树区间。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(n)。

## JavaScript 题解

```js
function buildTree(preorder, inorder) {
  const pos = new Map(inorder.map((v, i) => [v, i]));
  const build = (pl, pr, il, ir) => {
    if (pl > pr) return null;
    const val = preorder[pl], k = pos.get(val), left = k - il;
    return {
      val, left: build(pl + 1, pl + left, il, k-1), right: build(pl + left + 1, pr, k + 1, ir)
    };
  };
  return build(0, preorder.length-1, 0, inorder.length-1);
}
```
