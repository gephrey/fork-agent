# 094. 课程表

## 题目

给定课程依赖，判断能否完成所有课程。

## 最优思路

拓扑排序。统计入度，入度为 0 的课程入队；每学完一门课，降低后继入度。最终学完数量等于总数则无环。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(V+E)，空间 O(V+E)。

## JavaScript 题解

```js
function canFinish(numCourses, prerequisites) {
  const g = Array.from({
    length:numCourses
  }, () => []), indeg = Array(numCourses).fill(0);
  for (const[a, b] of prerequisites) {
    g[b].push(a);
    indeg[a]++;
  }
  const q = [];
  indeg.forEach((d, i) => {
    if (d === 0) q.push(i);
  });
  let cnt = 0;
  for (let h = 0;h < q.length;h++) {
    const u = q[h];
    cnt++;
    for (const v of g[u]) if (--indeg[v] === 0) q.push(v);
  }
  return cnt === numCourses;
}
```
