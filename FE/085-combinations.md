# 085. 组合

## 题目

返回 1..n 中所有 k 个数的组合。

## 最优思路

回溯从 start 开始递增选择，保证组合不重复。可用剩余数量剪枝。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(k*C(n,k))，空间 O(k)。

## JavaScript 题解

```js
function combine(n, k) {
  const ans = [], path = [];
  const dfs = start => {
    if (path.length === k) return ans.push([...path]);
    for (let i = start; i <= n-(k-path.length) + 1; i++) {
      path.push(i);
      dfs(i + 1);
      path.pop();
    }
  };
  dfs(1);
  return ans;
}
```
