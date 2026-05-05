# 017. 最长连续序列

## 题目

给定未排序数组，找最长连续整数序列长度。

## 最优思路

用 Set 存所有数。只从序列起点 x 开始扩展，即 x-1 不存在时才向后数，这样每个数最多被访问一次。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(n)。

## JavaScript 题解

```js
function longestConsecutive(nums) {
  const set = new Set(nums);
  let ans = 0;
  for (const x of set) {
    if (set.has(x - 1)) continue;
    let y = x;
    while (set.has(y)) y++;
    ans = Math.max(ans, y - x);
  }
  return ans;
}
```
