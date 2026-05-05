# 072. 将有序数组转换为二叉搜索树

## 题目

将升序数组转换为高度平衡 BST。

## 最优思路

每次选中点作为根，左半构建左子树，右半构建右子树。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(log n)。

## JavaScript 题解

```js
function sortedArrayToBST(nums) {
  const build = (l, r) => {
    if (l > r) return null;
    const m = (l + r) >> 1;
    return {
      val: nums[m], left: build(l, m-1), right: build(m + 1, r)
    };
  };
  return build(0, nums.length - 1);
}
```
