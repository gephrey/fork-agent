# 004. 最接近的三数之和

## 题目

给定数组 nums 和 target，返回最接近 target 的三数之和。

## 最优思路

排序后枚举第一个数，剩余两个数用双指针。每得到一个 sum 就更新当前最优；sum 小了左指针右移，大了右指针左移。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n^2)，空间 O(1)。

## JavaScript 题解

```js
function threeSumClosest(nums, target) {
  nums.sort((a, b) => a - b);
  let best = nums[0] + nums[1] + nums[2];
  for (let i = 0; i < nums.length - 2; i++) {
    let l = i + 1, r = nums.length - 1;
    while (l < r) {
      const sum = nums[i] + nums[l] + nums[r];
      if (Math.abs(sum - target) < Math.abs(best - target)) best = sum;
      if (sum === target) return sum;
      if (sum < target) l++;
else r--;
    }
  }
  return best;
}
```
