# 029. 字符串相乘

## 题目

两个非负整数字符串相乘，不能直接转数字。

## 最优思路

模拟竖式乘法。num1[i] * num2[j] 贡献到 res[i+j+1]，处理进位到 i+j。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(mn)，空间 O(m+n)。

## JavaScript 题解

```js
function multiply(a, b) {
  if (a === '0' || b === '0') return '0';
  const res = Array(a.length + b.length).fill(0);
  for (let i = a.length - 1; i >= 0; i--) {
    for (let j = b.length - 1; j >= 0; j--) {
      const sum = res[i + j + 1] + (a.charCodeAt(i) - 48) * (b.charCodeAt(j) - 48);
      res[i + j + 1] = sum % 10;
      res[i + j] += Math.floor(sum / 10);
    }
  }
  return res.join('').replace(/^0 + /, '');
}
```
