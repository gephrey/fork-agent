# 033. 验证回文串

## 题目

只考虑字母数字，忽略大小写，判断是否回文。

## 最优思路

双指针从两端向中间移动，跳过非字母数字字符，比较小写后的字符。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(1)。

## JavaScript 题解

```js
function isPalindrome(s) {
  let l = 0, r = s.length - 1;
  const ok = c => /[a-z0-9]/i.test(c);
  while (l < r) {
    while (l < r && !ok(s[l])) l++;
    while (l < r && !ok(s[r])) r--;
    if (s[l].toLowerCase() !== s[r].toLowerCase()) return false;
    l++;
    r--;
  }
  return true;
}
```
