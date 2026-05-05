---
name: fork-agent-note-writer
description: Use when writing, adding, or continuing study notes for the fork-agent project, especially requests like "/skill 增加或续写这一章的内容的笔记 /path/to/material". Notes must be written under /Users/gephrey/fork-agent/NOTES.
---

# Fork Agent Note Writer

Use this skill when the user asks to write, add, summarize, or continue notes for this project.

## Target Directory

All notes created or updated by this skill must live in:

```text
/Users/gephrey/fork-agent/NOTES
```

Do not write these notes into the source material directory unless the user explicitly asks for that.

## Invocation Pattern

The expected user pattern is:

```text
/skill 增加或续写这一章的内容的笔记 /xx/xx/xxx
```

Interpret it as:

- The text after `/skill` is the note-writing task.
- The final path is the source material, chapter, file, or directory to read.
- If the path is relative, resolve it from `/Users/gephrey/fork-agent`.
- If the source path points to a directory, inspect the most relevant files in that directory before writing.

## Workflow

1. Read the source material enough to understand the chapter or topic.
2. Check `/Users/gephrey/fork-agent/NOTES` for an existing related Markdown file.
3. If a related note exists, append or revise the relevant section without rewriting unrelated content.
4. If no related note exists, create a clearly named Markdown file in `/Users/gephrey/fork-agent/NOTES`.
5. Keep notes in Chinese by default. Preserve English API names, class names, package names, and code identifiers.
6. Prefer concise structured notes over raw excerpts.

## Note Shape

Use this structure when creating a new note unless the existing notes suggest a better local pattern:

```markdown
# 主题名

## 这章在讲什么

## 核心概念

## 实现流程

## 关键 API / 代码点

## 易错点

## 复习问题
```

For short topics, merge sections instead of padding.

## Quality Bar

- Capture the idea, flow, and why it matters.
- Include concrete API names or file names when useful.
- Avoid copying long source passages.
- Avoid turning notes into a code dump.
- If source material is code, explain the control flow and important abstractions.
- If source material is incomplete or ambiguous, state that in the note briefly.
