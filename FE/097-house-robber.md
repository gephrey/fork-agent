# 097. 打家劫舍

## 题目

不能偷相邻房屋，求最大金额。

## 最优思路

DP。对每间房，要么不偷继承 prev1，要么偷当前加 prev2，取最大。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(1)。

## JavaScript 题解

```js
function rob(nums) {
  let prev2 = 0, prev1 = 0;
  for (const x of nums) {
    [prev2, prev1] = [prev1, Math.max(prev1, prev2 + x)];
  }
  return prev1;
}
```
