# Copy With File And Line

VS Code extension that copies the active editor selection as:

```text
context: /absolute/path/to/file, L21~L45
```

## Usage

Select text in an editor and press `Command+Shift+C`.

The copied value contains:

- Absolute file path
- Selection start line
- Selection end line

## Development

```bash
pnpm install
pnpm run compile
```

Open this folder in VS Code, press `F5`, then test the command in the Extension Development Host.
