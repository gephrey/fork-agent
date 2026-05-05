import * as vscode from 'vscode';
import { buildClipboardText } from './copyText';

export function activate(context: vscode.ExtensionContext) {
  const disposable = vscode.commands.registerCommand('copyWithFileAndLine.copy', async () => {
    const editor = vscode.window.activeTextEditor;

    if (!editor) {
      vscode.window.showWarningMessage('No active editor.');
      return;
    }

    const filePath = editor.document.uri.fsPath;
    const text = buildClipboardText(filePath, editor.selection);

    await vscode.env.clipboard.writeText(text);
    vscode.window.setStatusBarMessage(`Copied: ${text}`, 3000);
    vscode.window.showInformationMessage('Copied file context to clipboard.');
  });

  context.subscriptions.push(disposable);
}

export function deactivate() {}
