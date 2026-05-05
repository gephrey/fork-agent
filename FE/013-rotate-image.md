# 013. 旋转图像

## 题目

将 n x n 矩阵原地顺时针旋转 90 度。

## 最优思路

先沿主对角线转置，再反转每一行。转置把 (i,j) 和 (j,i) 交换，行反转完成顺时针旋转。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n^2)，空间 O(1)。

## JavaScript 题解

```js
function rotate(matrix) {
  const n = matrix.length;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++)[matrix[i][j], matrix[j][i]] = [matrix[j][i], matrix[i][j]];
  }
  for (const row of matrix) row.reverse();
}
```
