# 074. 从中序与后序遍历构造二叉树

## 题目

给定 inorder 和 postorder，构造原二叉树。

## 最优思路

后序最后一个是根。通过根在中序中的位置拆左右子树，再递归处理对应区间。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(n)。

## JavaScript 题解

```js
function buildTree(inorder, postorder) {
  const pos = new Map(inorder.map((v, i) => [v, i]));
  const build = (il, ir, pl, pr) => {
    if (il > ir) return null;
    const val = postorder[pr], k = pos.get(val), left = k - il;
    return {
      val, left: build(il, k-1, pl, pl + left-1), right: build(k + 1, ir, pl + left, pr-1)
    };
  };
  return build(0, inorder.length-1, 0, postorder.length-1);
}
```
