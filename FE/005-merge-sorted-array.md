# 005. 合并两个有序数组

## 题目

nums1 有足够空间容纳 nums2，将 nums2 合并进 nums1，保持有序。

## 最优思路

从后往前合并。比较 nums1 有效尾部和 nums2 尾部，把较大值放到 nums1 的最后空位，避免覆盖 nums1 中尚未比较的元素。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(m+n)，空间 O(1)。

## JavaScript 题解

```js
function merge(nums1, m, nums2, n) {
  let i = m - 1, j = n - 1, k = m + n - 1;
  while (j >= 0) {
    if (i >= 0 && nums1[i] > nums2[j]) nums1[k--] = nums1[i--];
else nums1[k--] = nums2[j--];
  }
}
```
