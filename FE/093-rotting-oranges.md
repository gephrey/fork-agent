# 093. 腐烂的橘子

## 题目

每分钟腐烂橘子感染四邻新鲜橘子，求全部腐烂最短时间。

## 最优思路

多源 BFS。所有腐烂橘子同时入队，按层扩散，每扩散一层时间加一。最后若仍有新鲜橘子返回 -1。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(mn)，空间 O(mn)。

## JavaScript 题解

```js
function orangesRotting(grid) {
  const m = grid.length, n = grid[0].length, q = [];
  let fresh = 0;
  for (let i = 0;i < m;i++) for (let j = 0;j < n;j++) {
    if (grid[i][j] === 2) q.push([i, j]);
    if (grid[i][j] === 1) fresh++;
  }
  let time = 0, head = 0, dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]];
  while (head < q.length && fresh) {
    const size = q.length-head;
    for (let s = 0;s < size;s++) {
      const[i, j] = q[head++];
      for (const[di, dj] of dirs) {
        const x = i + di, y = j + dj;
        if (x >= 0 && x < m && y >= 0 && y < n && grid[x][y] === 1) {
          grid[x][y] = 2;
          fresh--;
          q.push([x, y]);
        }
      }
    }
    time++;
  }
  return fresh ? -1: time;
}
```
