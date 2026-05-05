# 088. 括号生成

## 题目

生成 n 对括号的所有合法组合。

## 最优思路

回溯维护已用左括号和右括号数量。左括号小于 n 可放左括号；右括号少于左括号才可放右括号。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(Catalan(n)*n)，空间 O(n)。

## JavaScript 题解

```js
function generateParenthesis(n) {
  const ans = [];
  const dfs = (s, l, r) => {
    if (s.length === 2 * n) return ans.push(s);
    if (l < n) dfs(s + '(', l + 1, r);
    if (r < l) dfs(s + ')', l, r + 1);
  };
  dfs('', 0, 0);
  return ans;
}
```
