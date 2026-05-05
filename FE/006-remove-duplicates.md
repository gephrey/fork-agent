# 006. 删除有序数组中的重复项

## 题目

原地删除有序数组重复元素，返回去重后的长度。

## 最优思路

快慢指针。slow 指向当前去重数组的最后一位，fast 扫描新值；当 nums[fast] 与 nums[slow] 不同时，把新值放到++slow。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(1)。

## JavaScript 题解

```js
function removeDuplicates(nums) {
  if (nums.length === 0) return 0;
  let slow = 0;
  for (let fast = 1; fast < nums.length; fast++) {
    if (nums[fast] !== nums[slow]) nums[++slow] = nums[fast];
  }
  return slow + 1;
}
```
