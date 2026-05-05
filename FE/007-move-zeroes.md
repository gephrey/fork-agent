# 007. 移动零

## 题目

将数组中的 0 移动到末尾，同时保持非零元素相对顺序。

## 最优思路

用 slow 指向下一个非零元素应该放置的位置。先把所有非零元素依次前移，再把 slow 之后的位置填 0。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(1)。

## JavaScript 题解

```js
function moveZeroes(nums) {
  let slow = 0;
  for (const x of nums) if (x !== 0) nums[slow++] = x;
  while (slow < nums.length) nums[slow++] = 0;
}
```
