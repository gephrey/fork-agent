# 019. 数组中的第 K 个最大元素

## 题目

返回数组中第 k 大的元素。

## 最优思路

快速选择。目标下标是 n-k。每次 partition 后只递归目标所在一侧，平均线性时间。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

平均时间 O(n)，最坏 O(n^2)，空间 O(1)。

## JavaScript 题解

```js
function findKthLargest(nums, k) {
  const target = nums.length - k;
  let l = 0, r = nums.length - 1;
  while (true) {
    const p = partition(nums, l, r);
    if (p === target) return nums[p];
    if (p < target) l = p + 1;
else r = p - 1;
  }
}
function partition(a, l, r) {
  const pivot = a[r];
  let i = l;
  for (let j = l; j < r; j++) if (a[j] <= pivot)[a[i++], a[j]] = [a[j], a[i]];
  [a[i], a[r]] = [a[r], a[i]];
  return i;
}
```
