# 087. 组合总和 II

## 题目

候选数每个只能用一次，返回和为 target 的不重复组合。

## 最优思路

排序后回溯。同层重复值跳过；下一层从 i+1 开始，表示每个元素只能用一次。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间指数级，空间 O(n)。

## JavaScript 题解

```js
function combinationSum2(candidates, target) {
  candidates.sort((a, b) => a-b);
  const ans = [], path = [];
  const dfs = (start, rest) => {
    if (rest === 0) return ans.push([...path]);
    for (let i = start;i < candidates.length && candidates[i] <= rest;i++) {
      if (i > start && candidates[i] === candidates[i-1]) continue;
      path.push(candidates[i]);
      dfs(i + 1, rest-candidates[i]);
      path.pop();
    }
  };
  dfs(0, target);
  return ans;
}
```
