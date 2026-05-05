# 076. 二叉树的序列化与反序列化

## 题目

把二叉树转成字符串，再从字符串还原。

## 最优思路

前序序列化，空节点写 #。反序列化时按同样顺序消费 token，遇 # 返回 null。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(n)。

## JavaScript 题解

```js
function serialize(root) {
  const arr = [];
  const dfs = n => {
    if (!n) return arr.push('#');
    arr.push(String(n.val));
    dfs(n.left);
    dfs(n.right);
  };
  dfs(root);
  return arr.join(', ');
}
function deserialize(data) {
  const arr = data.split(', ');
  let i = 0;
  const dfs = () => {
    const v = arr[i++];
    if (v === '#') return null;
    const node = {
      val: Number(v), left: null, right: null
    };
    node.left = dfs();
    node.right = dfs();
    return node;
  };
  return dfs();
}
```
