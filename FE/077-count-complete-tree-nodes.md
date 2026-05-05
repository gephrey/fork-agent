# 077. 完全二叉树的节点个数

## 题目

统计完全二叉树节点数。

## 最优思路

利用完全二叉树性质。若左高等于右高，说明左子树满；否则右子树满，递归另一边。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(log^2 n)，空间 O(log n)。

## JavaScript 题解

```js
function countNodes(root) {
  if (!root) return 0;
  const height = n => {
    let h = 0;
    while (n) {
      h++;
      n = n.left;
    }
    return h;
  };
  const lh = height(root.left), rh = height(root.right);
  if (lh === rh) return(1 << lh) + countNodes(root.right);
  return(1 << rh) + countNodes(root.left);
}
```
