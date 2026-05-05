# 086. 组合总和

## 题目

给定无重复候选数，可重复使用，找和为 target 的组合。

## 最优思路

排序后回溯。选择当前数后下一层仍从 i 开始，表示可重复使用；超过 target 剪枝。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间指数级，空间 O(target/min)。

## JavaScript 题解

```js
function combinationSum(candidates, target) {
  candidates.sort((a, b) => a-b);
  const ans = [], path = [];
  const dfs = (start, rest) => {
    if (rest === 0) return ans.push([...path]);
    for (let i = start;i < candidates.length && candidates[i] <= rest;i++) {
      path.push(candidates[i]);
      dfs(i, rest-candidates[i]);
      path.pop();
    }
  };
  dfs(0, target);
  return ans;
}
```
