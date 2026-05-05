# 012. 螺旋矩阵

## 题目

按顺时针螺旋顺序返回矩阵所有元素。

## 最优思路

维护上下左右四个边界，每轮依次走上边、右边、下边、左边，走完后收缩边界。每步前检查边界是否仍有效。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(mn)，空间 O(1)，不计答案。

## JavaScript 题解

```js
function spiralOrder(matrix) {
  const ans = [];
  let top = 0, bottom = matrix.length - 1, left = 0, right = matrix[0].length - 1;
  while (top <= bottom && left <= right) {
    for (let j = left; j <= right; j++) ans.push(matrix[top][j]);
    top++;
    for (let i = top; i <= bottom; i++) ans.push(matrix[i][right]);
    right--;
    if (top <= bottom) for (let j = right; j >= left; j--) ans.push(matrix[bottom][j]);
    bottom--;
    if (left <= right) for (let i = bottom; i >= top; i--) ans.push(matrix[i][left]);
    left++;
  }
  return ans;
}
```
