"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;
exports.deactivate = deactivate;
const vscode = require("vscode");
const installer_1 = require("./installer");
async function activate(context) {
    console.log('Tafkir VS Code Extension activated');
    // Automatically install the kernel on activation if it doesn't exist or is outdated
    try {
        await (0, installer_1.installKernel)(context);
    }
    catch (error) {
        console.error('Failed to install Tafkir Jupyter Kernel on activation:', error);
    }
    // Command to manually reinstall
    let disposable = vscode.commands.registerCommand('tafkir.reinstallKernel', async () => {
        try {
            await (0, installer_1.installKernel)(context, true);
            vscode.window.showInformationMessage('Tafkir Jupyter Kernel successfully reinstalled!');
        }
        catch (error) {
            vscode.window.showErrorMessage(`Failed to install kernel: ${error.message}`);
        }
    });
    context.subscriptions.push(disposable);
}
function deactivate() { }
//# sourceMappingURL=extension.js.map