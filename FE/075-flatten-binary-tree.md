# 075. 二叉树展开为链表

## 题目

将二叉树原地展开为前序链表。

## 最优思路

后序处理。先展开左右子树，再把左子树接到右侧，并把原右子树接到左子树最右端。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(h)。

## JavaScript 题解

```js
function flatten(root) {
  if (!root) return;
  flatten(root.left);
  flatten(root.right);
  const right = root.right;
  root.right = root.left;
  root.left = null;
  let p = root;
  while (p.right) p = p.right;
  p.right = right;
}
```
