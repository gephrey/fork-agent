# 024. 有效的括号

## 题目

判断括号字符串是否合法。

## 最优思路

栈保存待匹配的左括号。遇到右括号时检查栈顶是否匹配，不匹配直接 false。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(n)。

## JavaScript 题解

```js
function isValid(s) {
  const stack = [];
  const pair = {
    ')': '(', ']': '[', '}': '{'
  };
  for (const ch of s) {
    if (ch === '(' || ch === '[' || ch === '{') stack.push(ch);
else if (stack.pop() !== pair[ch]) return false;
  }
  return stack.length === 0;
}
```
