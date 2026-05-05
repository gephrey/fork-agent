# 025. 最小覆盖子串

## 题目

在 s 中找包含 t 所有字符的最短子串。

## 最优思路

滑动窗口。need 记录需求，右指针扩张直到满足，再不断收缩左指针并更新答案。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(k)。

## JavaScript 题解

```js
function minWindow(s, t) {
  const need = new Map();
  for (const c of t) need.set(c, (need.get(c) || 0) + 1);
  let missing = t.length, l = 0, bestL = 0, best = Infinity;
  for (let r = 0; r < s.length; r++) {
    if ((need.get(s[r]) || 0) > 0) missing--;
    need.set(s[r], (need.get(s[r]) || 0) - 1);
    while (missing === 0) {
      if (r - l + 1 < best) {
        best = r - l + 1;
        bestL = l;
      }
      need.set(s[l], (need.get(s[l]) || 0) + 1);
      if (need.get(s[l]) > 0) missing++;
      l++;
    }
  }
  return best === Infinity ? '': s.slice(bestL, bestL + best);
}
```
