"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.installKernel = installKernel;
const path = require("path");
const fs = require("fs");
const os = require("os");
async function installKernel(context, force = false) {
    // Determine Jupyter data dir
    let jupyterDataDir = '';
    if (process.platform === 'win32') {
        jupyterDataDir = path.join(process.env.APPDATA || '', 'jupyter');
    }
    else if (process.platform === 'darwin') {
        jupyterDataDir = path.join(os.homedir(), 'Library', 'Jupyter');
    }
    else {
        jupyterDataDir = path.join(os.homedir(), '.local', 'share', 'jupyter');
    }
    const kernelDir = path.join(jupyterDataDir, 'kernels', 'tafkir');
    const targetJarPath = path.join(kernelDir, 'tafkir-kernel.jar');
    const kernelJsonPath = path.join(kernelDir, 'kernel.json');
    const sourceJarPath = path.join(context.extensionPath, 'resources', 'tafkir-kernel.jar');
    if (!fs.existsSync(sourceJarPath)) {
        throw new Error(`Kernel JAR not found in extension package: ${sourceJarPath}`);
    }
    // Create directories
    if (!fs.existsSync(kernelDir)) {
        fs.mkdirSync(kernelDir, { recursive: true });
    }
    // Copy JAR
    fs.copyFileSync(sourceJarPath, targetJarPath);
    // Write kernel.json
    const kernelJson = {
        "argv": [
            "java",
            "--enable-preview",
            "--add-modules=jdk.incubator.vector",
            "--add-opens=jdk.jshell/jdk.jshell=ALL-UNNAMED",
            "--enable-native-access=ALL-UNNAMED",
            "-Xmx4g",
            "-XX:+UseG1GC",
            "-jar",
            targetJarPath,
            "{connection_file}"
        ],
        "display_name": "Tafkir (Java 25 + AI/ML)",
        "language": "java",
        "interrupt_mode": "message",
        "env": {},
        "metadata": {
            "debugger": false
        }
    };
    fs.writeFileSync(kernelJsonPath, JSON.stringify(kernelJson, null, 2));
    console.log(`Successfully installed Tafkir kernel to ${kernelDir}`);
}
//# sourceMappingURL=installer.js.map