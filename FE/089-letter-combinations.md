# 089. 电话号码的字母组合

## 题目

返回数字字符串对应的所有字母组合。

## 最优思路

回溯按位选择。每一层处理一个数字，从映射表里枚举所有字母。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(4^n*n)，空间 O(n)。

## JavaScript 题解

```js
function letterCombinations(digits) {
  if (!digits) return [];
  const map = {
    2:'abc', 3:'def', 4:'ghi', 5:'jkl', 6:'mno', 7:'pqrs', 8:'tuv', 9:'wxyz'
  };
  const ans = [];
  const dfs = (i, s) => {
    if (i === digits.length) return ans.push(s);
    for (const c of map[digits[i]]) dfs(i + 1, s + c);
  };
  dfs(0, '');
  return ans;
}
```
