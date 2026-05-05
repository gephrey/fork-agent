# 011. 合并区间

## 题目

给定若干区间，合并所有重叠区间。

## 最优思路

按左端点排序，依次扫描。当前区间与结果最后一个区间重叠则扩展右端点，否则加入结果。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n log n)，空间 O(n)。

## JavaScript 题解

```js
function merge(intervals) {
  intervals.sort((a, b) => a[0] - b[0]);
  const ans = [];
  for (const cur of intervals) {
    const last = ans[ans.length - 1];
    if (!last || cur[0] > last[1]) ans.push(cur);
else last[1] = Math.max(last[1], cur[1]);
  }
  return ans;
}
```
