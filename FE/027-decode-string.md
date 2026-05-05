# 027. 字符串解码

## 题目

解码形如 3[a2[c]] 的字符串。

## 最优思路

栈保存进入括号前的字符串和重复次数。遇 [ 入栈并清空当前串，遇 ] 出栈拼接。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n + 输出长度)，空间 O(n)。

## JavaScript 题解

```js
function decodeString(s) {
  const stack = [];
  let num = 0, cur = '';
  for (const ch of s) {
    if (/\d/.test(ch)) num = num * 10 + Number(ch);
else if (ch === '[') {
      stack.push([cur, num]);
      cur = '';
      num = 0;
    } else if (ch === ']') {
      const[pre, k] = stack.pop();
      cur = pre + cur.repeat(k);
    } else cur += ch;
  }
  return cur;
}
```
