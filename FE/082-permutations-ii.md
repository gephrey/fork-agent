# 082. 全排列 II

## 题目

返回可能包含重复数字数组的不重复全排列。

## 最优思路

排序后回溯。若当前值等于前一个值且前一个值本层还没被使用，则跳过，避免同层重复选择。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n*n!)，空间 O(n)。

## JavaScript 题解

```js
function permuteUnique(nums) {
  nums.sort((a, b) => a - b);
  const ans = [], used = Array(nums.length).fill(false), path = [];
  const dfs = () => {
    if (path.length === nums.length) return ans.push([...path]);
    for (let i = 0; i < nums.length; i++) {
      if (used[i] || (i > 0 && nums[i] === nums[i - 1] && !used[i - 1])) continue;
      used[i] = true;
      path.push(nums[i]);
      dfs();
      path.pop();
      used[i] = false;
    }
  };
  dfs();
  return ans;
}
```
