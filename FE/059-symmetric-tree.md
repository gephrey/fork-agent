# 059. 对称二叉树

## 题目

判断二叉树是否镜像对称。

## 最优思路

递归比较两棵子树：外侧对外侧，内侧对内侧，值也要相等。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(h)。

## JavaScript 题解

```js
function isSymmetric(root) {
  const same = (a, b) => {
    if (!a && !b) return true;
    if (!a || !b) return false;
    return a.val === b.val && same(a.left, b.right) && same(a.right, b.left);
  };
  return same(root.left, root.right);
}
```
