import * as vscode from 'vscode';
import { installKernel } from './installer';

export async function activate(context: vscode.ExtensionContext) {
    console.log('Tafkir VS Code Extension activated');

    // Automatically install the kernel on activation if it doesn't exist or is outdated
    try {
        await installKernel(context);
    } catch (error) {
        console.error('Failed to install Tafkir Jupyter Kernel on activation:', error);
    }

    // Command to manually reinstall
    let disposable = vscode.commands.registerCommand('tafkir.reinstallKernel', async () => {
        try {
            await installKernel(context, true);
            vscode.window.showInformationMessage('Tafkir Jupyter Kernel successfully reinstalled!');
        } catch (error: any) {
            vscode.window.showErrorMessage(`Failed to install kernel: ${error.message}`);
        }
    });

    context.subscriptions.push(disposable);
}

export function deactivate() {}
