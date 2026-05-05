# 008. 颜色分类

## 题目

只包含 0、1、2 的数组，原地按 0、1、2 排序。

## 最优思路

荷兰国旗算法。p0 左侧全是 0，p2 右侧全是 2，i 扫描中间未知区；遇 0 换到左边，遇 2 换到右边且 i 不前进。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(1)。

## JavaScript 题解

```js
function sortColors(nums) {
  let p0 = 0, i = 0, p2 = nums.length - 1;
  while (i <= p2) {
    if (nums[i] === 0)[nums[i++], nums[p0++]] = [nums[p0], nums[i]];
else if (nums[i] === 2)[nums[i], nums[p2--]] = [nums[p2], nums[i]];
else i++;
  }
}
```
