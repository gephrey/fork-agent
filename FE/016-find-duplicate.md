# 016. 寻找重复数

## 题目

长度 n+1 的数组，值在 1..n，只有一个重复数，不能修改数组。

## 最优思路

把数组看成链表：下标 i 指向 nums[i]，重复值会造成环。用快慢指针找环入口，入口就是重复数。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(1)。

## JavaScript 题解

```js
function findDuplicate(nums) {
  let slow = nums[0], fast = nums[nums[0]];
  while (slow !== fast) {
    slow = nums[slow];
    fast = nums[nums[fast]];
  }
  slow = 0;
  while (slow !== fast) {
    slow = nums[slow];
    fast = nums[fast];
  }
  return slow;
}
```
