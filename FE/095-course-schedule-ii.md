# 095. 课程表 II

## 题目

返回一种可完成所有课程的学习顺序，若不可能返回空数组。

## 最优思路

拓扑排序并记录出队顺序。若最终顺序长度不足 numCourses，说明存在环。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(V+E)，空间 O(V+E)。

## JavaScript 题解

```js
function findOrder(numCourses, prerequisites) {
  const g = Array.from({
    length:numCourses
  }, () => []), indeg = Array(numCourses).fill(0);
  for (const[a, b] of prerequisites) {
    g[b].push(a);
    indeg[a]++;
  }
  const q = [], ans = [];
  indeg.forEach((d, i) => {
    if (d === 0) q.push(i);
  });
  for (let h = 0;h < q.length;h++) {
    const u = q[h];
    ans.push(u);
    for (const v of g[u]) if (--indeg[v] === 0) q.push(v);
  }
  return ans.length === numCourses ? ans:[];
}
```
