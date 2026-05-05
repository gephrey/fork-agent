# 098. 最大子数组和

## 题目

找连续子数组的最大和。

## 最优思路

Kadane 算法。cur 表示以当前元素结尾的最大和，要么接前面，要么从当前重新开始。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(1)。

## JavaScript 题解

```js
function maxSubArray(nums) {
  let cur = nums[0], ans = nums[0];
  for (let i = 1;i < nums.length;i++) {
    cur = Math.max(nums[i], cur + nums[i]);
    ans = Math.max(ans, cur);
  }
  return ans;
}
```
