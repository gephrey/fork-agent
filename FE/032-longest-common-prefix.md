# 032. 最长公共前缀

## 题目

返回字符串数组的最长公共前缀。

## 最优思路

以第一个字符串为基准逐位比较，任意字符串越界或字符不同则返回当前前缀。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(总字符数)，空间 O(1)。

## JavaScript 题解

```js
function longestCommonPrefix(strs) {
  if (!strs.length) return '';
  for (let i = 0; i < strs[0].length; i++) {
    for (const s of strs) {
      if (i === s.length || s[i] !== strs[0][i]) return strs[0].slice(0, i);
    }
  }
  return strs[0];
}
```
