# 080. 填充每个节点的下一个右侧节点指针

## 题目

给每个节点 next 指向同层右侧节点，没有则 null。

## 最优思路

层序遍历最通用。每层遍历时把前一个节点的 next 指向当前节点，层末尾 next 为 null。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(n)。

## JavaScript 题解

```js
function connect(root) {
  if (!root) return root;
  const q = [root];
  for (let head = 0; head < q.length;) {
    const size = q.length - head;
    let prev = null;
    for (let i = 0; i < size; i++) {
      const node = q[head++];
      if (prev) prev.next = node;
      prev = node;
      if (node.left) q.push(node.left);
      if (node.right) q.push(node.right);
    }
    prev.next = null;
  }
  return root;
}
```
