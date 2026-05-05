# 028. 字符串相加

## 题目

两个非负整数字符串相加，不能直接转数字。

## 最优思路

从低位到高位逐位相加，维护进位 carry，最后反转结果。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(max(m,n))，空间 O(max(m,n))。

## JavaScript 题解

```js
function addStrings(a, b) {
  let i = a.length - 1, j = b.length - 1, carry = 0, res = '';
  while (i >= 0 || j >= 0 || carry) {
    const x = i >= 0 ? a.charCodeAt(i--) - 48: 0;
    const y = j >= 0 ? b.charCodeAt(j--) - 48: 0;
    const sum = x + y + carry;
    res = String(sum % 10) + res;
    carry = Math.floor(sum / 10);
  }
  return res;
}
```
