# 081. 全排列

## 题目

返回不含重复数字数组的所有排列。

## 最优思路

回溯选择每个未使用数字。路径长度等于 n 时加入答案。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n*n!)，空间 O(n)。

## JavaScript 题解

```js
function permute(nums) {
  const ans = [], used = Array(nums.length).fill(false), path = [];
  const dfs = () => {
    if (path.length === nums.length) return ans.push([...path]);
    for (let i = 0; i < nums.length; i++) if (!used[i]) {
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
