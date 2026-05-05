# 031. 比较版本号

## 题目

比较 version1 和 version2 的大小。

## 最优思路

按点分割，逐段转数字比较。缺失段按 0 处理。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n+m)，空间 O(n+m)。

## JavaScript 题解

```js
function compareVersion(a, b) {
  const x = a.split('.'), y = b.split('.');
  const n = Math.max(x.length, y.length);
  for (let i = 0; i < n; i++) {
    const p = Number(x[i] || 0), q = Number(y[i] || 0);
    if (p !== q) return p > q ? 1: -1;
  }
  return 0;
}
```
