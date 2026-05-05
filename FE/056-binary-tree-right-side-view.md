# 056. 二叉树的右视图

## 题目

返回从右侧能看到的节点值。

## 最优思路

层序遍历时每层最后一个节点就是右视图。也可以 DFS 先右后左，第一次到达某深度的节点就是答案。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(n)。

## JavaScript 题解

```js
function rightSideView(root) {
  if (!root) return [];
  const q = [root], ans = [];
  for (let head = 0; head < q.length;) {
    const size = q.length - head;
    for (let i = 0; i < size; i++) {
      const node = q[head++];
      if (i === size - 1) ans.push(node.val);
      if (node.left) q.push(node.left);
      if (node.right) q.push(node.right);
    }
  }
  return ans;
}
```
