# 023. 回文子串

## 题目

统计字符串中回文子串数量。

## 最优思路

中心扩展。每个位置作为奇数中心，每两个位置之间作为偶数中心，扩展成功一次就计数一次。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n^2)，空间 O(1)。

## JavaScript 题解

```js
function countSubstrings(s) {
  let ans = 0;
  const expand = (l, r) => {
    while (l >= 0 && r < s.length && s[l] === s[r]) {
      ans++;
      l--;
      r++;
    }
  };
  for (let i = 0; i < s.length; i++) {
    expand(i, i);
    expand(i, i + 1);
  }
  return ans;
}
```
