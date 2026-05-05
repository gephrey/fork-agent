# 020. 除自身以外数组的乘积

## 题目

返回数组 output，output[i] 等于除 nums[i] 外其他元素乘积，不能用除法。

## 最优思路

两趟扫描。第一趟把左侧乘积写入 ans[i]，第二趟维护右侧乘积并乘到 ans[i] 上。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，额外空间 O(1)，不计答案。

## JavaScript 题解

```js
function productExceptSelf(nums) {
  const ans = Array(nums.length).fill(1);
  let left = 1;
  for (let i = 0; i < nums.length; i++) {
    ans[i] = left;
    left *= nums[i];
  }
  let right = 1;
  for (let i = nums.length - 1; i >= 0; i--) {
    ans[i] *= right;
    right *= nums[i];
  }
  return ans;
}
```
