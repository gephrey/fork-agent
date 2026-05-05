export type PositionLike = {
  line: number;
  character: number;
};

export type SelectionLike = {
  start: PositionLike;
  end: PositionLike;
  isEmpty: boolean;
};

export function buildClipboardText(filePath: string, selection: SelectionLike): string {
  if (selection.isEmpty) {
    return filePath;
  }

  const startLine = selection.start.line + 1;
  const exclusiveEndAtLineStart = selection.end.character === 0;
  const endLine = Math.max(
    selection.start.line,
    selection.end.line - (exclusiveEndAtLineStart ? 1 : 0),
  ) + 1;

  return `context: ${filePath}, L${startLine}~L${endLine}`;
}
