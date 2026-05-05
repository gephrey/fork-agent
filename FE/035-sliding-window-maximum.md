# 035. 滑动窗口最大值

## 题目

返回每个大小为 k 的滑动窗口最大值。

## 最优思路

单调递减队列保存下标。队头是当前窗口最大值，下标过期则弹出，入队前弹出所有小于等于当前值的下标。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(k)。

## JavaScript 题解

```js
function maxSlidingWindow(nums, k) {
  const q = [], ans = [];
  for (let i = 0; i < nums.length; i++) {
    while (q.length && q[0] <= i - k) q.shift();
    while (q.length && nums[q[q.length-1]] <= nums[i]) q.pop();
    q.push(i);
    if (i >= k - 1) ans.push(nums[q[0]]);
  }
  return ans;
}
```
