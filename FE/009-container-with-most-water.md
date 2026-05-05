# 009. 盛最多水的容器

## 题目

给定高度数组，找两条线能容纳的最大水量。

## 最优思路

双指针从两端开始。面积由短板决定，移动较高的一侧不会让短板变高，只可能变窄，所以每次移动较短的一侧。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(1)。

## JavaScript 题解

```js
function maxArea(height) {
  let l = 0, r = height.length - 1, ans = 0;
  while (l < r) {
    ans = Math.max(ans, Math.min(height[l], height[r]) * (r - l));
    if (height[l] < height[r]) l++;
else r--;
  }
  return ans;
}
```
