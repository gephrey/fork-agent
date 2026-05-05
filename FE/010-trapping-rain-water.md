# 010. 接雨水

## 题目

给定柱子高度，计算能接住多少雨水。

## 最优思路

双指针维护 leftMax 和 rightMax。较低一侧的最大水位已经由该侧 max 决定，可以结算并移动；无需预处理数组。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(1)。

## JavaScript 题解

```js
function trap(height) {
  let l = 0, r = height.length - 1, lm = 0, rm = 0, ans = 0;
  while (l < r) {
    if (height[l] < height[r]) {
      lm = Math.max(lm, height[l]);
      ans += lm - height[l++];
    } else {
      rm = Math.max(rm, height[r]);
      ans += rm - height[r--];
    }
  }
  return ans;
}
```
