# 034. 字符串转换整数 atoi

## 题目

实现 atoi，处理空格、符号、数字和 32 位边界。

## 最优思路

跳过前导空格，读取符号，持续读数字并在每步做边界截断。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(1)。

## JavaScript 题解

```js
function myAtoi(s) {
  let i = 0, sign = 1, ans = 0;
  const MAX = 2147483647, MIN = -2147483648;
  while (s[i] === ' ') i++;
  if (s[i] === ' + ' || s[i] === '-') sign = s[i++] === '-' ? -1: 1;
  while (i < s.length && /d/.test(s[i])) {
    ans = ans * 10 + Number(s[i++]);
    if (sign * ans > MAX) return MAX;
    if (sign * ans < MIN) return MIN;
  }
  return sign * ans;
}
```
