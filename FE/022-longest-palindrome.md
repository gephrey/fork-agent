# 022. 最长回文子串

## 题目

返回字符串中最长的回文子串。

## 最优思路

枚举每个中心，分别扩展奇数长度和偶数长度回文。记录最长的左右边界。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n^2)，空间 O(1)。

## JavaScript 题解

```js
function longestPalindrome(s) {
  let start = 0, len = 0;
  const expand = (l, r) => {
    while (l >= 0 && r < s.length && s[l] === s[r]) {
      l--;
      r++;
    }
    if (r - l - 1 > len) {
      start = l + 1;
      len = r - l - 1;
    }
  };
  for (let i = 0; i < s.length; i++) {
    expand(i, i);
    expand(i, i + 1);
  }
  return s.slice(start, start + len);
}
```
