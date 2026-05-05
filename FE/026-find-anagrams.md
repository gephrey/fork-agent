# 026. 找到字符串中所有字母异位词

## 题目

在 s 中找到 p 的所有异位词起始下标。

## 最优思路

固定长度滑动窗口。维护需求计数和 missing，窗口超过 p 长度时移出左字符。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(1)。

## JavaScript 题解

```js
function findAnagrams(s, p) {
  const cnt = Array(26).fill(0), ans = [];
  for (const c of p) cnt[c.charCodeAt(0) - 97]++;
  let missing = p.length;
  for (let r = 0, l = 0; r < s.length; r++) {
    if (cnt[s.charCodeAt(r) - 97]--> 0) missing--;
    if (r - l + 1 > p.length &&++cnt[s.charCodeAt(l++) - 97] > 0) missing++;
    if (missing === 0) ans.push(l);
  }
  return ans;
}
```
