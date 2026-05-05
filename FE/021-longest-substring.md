# 021. 无重复字符的最长子串

## 题目

给定字符串 s，找不含重复字符的最长子串长度。

## 最优思路

滑动窗口维护当前无重复区间。右指针扩张，若字符重复就移动左指针到上次出现位置之后。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(k)。

## JavaScript 题解

```js
function lengthOfLongestSubstring(s) {
  const map = new Map();
  let l = 0, ans = 0;
  for (let r = 0; r < s.length; r++) {
    if (map.has(s[r]) && map.get(s[r]) >= l) l = map.get(s[r]) + 1;
    map.set(s[r], r);
    ans = Math.max(ans, r - l + 1);
  }
  return ans;
}
```
