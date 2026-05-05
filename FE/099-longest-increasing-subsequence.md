# 099. 最长递增子序列

## 题目

返回数组中最长严格递增子序列长度。

## 最优思路

贪心加二分。tails[len] 存长度为 len+1 的递增子序列的最小结尾；每个数替换第一个 >= 它的位置。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n log n)，空间 O(n)。

## JavaScript 题解

```js
function lengthOfLIS(nums) {
  const tails = [];
  for (const x of nums) {
    let l = 0, r = tails.length;
    while (l < r) {
      const m = (l + r) >> 1;
      if (tails[m] < x) l = m + 1;
else r = m;
    }
    tails[l] = x;
  }
  return tails.length;
}
```
