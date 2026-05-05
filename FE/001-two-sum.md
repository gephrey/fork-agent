# 001. 两数之和

## 题目

给定整数数组 nums 和目标值 target，找出和为 target 的两个数下标。

## 最优思路

哈希表一次遍历。遍历到 x 时，只需要判断 target - x 是否已出现；若出现直接返回两个下标，否则把 x 和当前下标放入表。这样避免双重循环。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(n)。

## JavaScript 题解

```js
function twoSum(nums, target) {
  const map = new Map();
  for (let i = 0; i < nums.length; i++) {
    const need = target - nums[i];
    if (map.has(need)) return [map.get(need), i];
    map.set(nums[i], i);
  }
  return [];
}
```
