# 084. 子集 II

## 题目

返回可能包含重复数字数组的所有不重复子集。

## 最优思路

先排序。回溯同一层遇到相同数字只使用第一个，跳过后续重复分支。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n*2^n)，空间 O(n)。

## JavaScript 题解

```js
function subsetsWithDup(nums) {
  nums.sort((a, b) => a-b);
  const ans = [], path = [];
  const dfs = start => {
    ans.push([...path]);
    for (let i = start;i < nums.length;i++) {
      if (i > start && nums[i] === nums[i-1]) continue;
      path.push(nums[i]);
      dfs(i + 1);
      path.pop();
    }
  };
  dfs(0);
  return ans;
}
```
