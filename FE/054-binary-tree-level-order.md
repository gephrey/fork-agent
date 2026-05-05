# 054. 二叉树的层序遍历

## 题目

按层返回二叉树节点值。

## 最优思路

BFS 队列。每轮记录当前队列长度，这一批就是同一层，处理完后进入下一层。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(n)。

## JavaScript 题解

```js
function levelOrder(root) {
  if (!root) return [];
  const q = [root], ans = [];
  for (let head = 0; head < q.length;) {
    const size = q.length - head, level = [];
    for (let i = 0; i < size; i++) {
      const node = q[head++];
      level.push(node.val);
      if (node.left) q.push(node.left);
      if (node.right) q.push(node.right);
    }
    ans.push(level);
  }
  return ans;
}
```
