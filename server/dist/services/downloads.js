import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
export const downloadProgress = new Map();
export function getLocalHuggingFaceModels() {
    const models = [];
    const projectDirs = [join(process.cwd(), 'models')];
    for (const projectDir of projectDirs) {
        if (!existsSync(projectDir))
            continue;
        try {
            const entries = readdirSync(projectDir, { withFileTypes: true });
            for (const entry of entries) {
                if (!entry.isDirectory())
                    continue;
                let modelId = entry.name;
                const modelMatch = entry.name.match(/^models--(.+)$/);
                if (modelMatch) {
                    modelId = modelMatch[1].replace(/--/g, '/');
                }
                const orgMatch = modelId.match(/^([^\/]+)\//);
                const ownedBy = orgMatch ? orgMatch[1] : 'local';
                if (!models.find(m => m.id === modelId)) {
                    models.push({ id: modelId, object: 'model', created: Math.floor(Date.now() / 1000), owned_by: ownedBy });
                }
            }
        }
        catch {
            continue;
        }
    }
    return models;
}
export async function downloadHuggingFaceModel(modelId) {
    const modelsDir = join(process.cwd(), 'models');
    const modelDir = join(modelsDir, `${modelId.replace('/', '--')}`);
    const progress = { model: modelId, status: 'downloading', progress: 0, message: 'Initializing download...', startTime: Date.now() };
    downloadProgress.set(modelId, progress);
    try {
        if (!existsSync(modelsDir)) {
            mkdirSync(modelsDir, { recursive: true });
        }
        if (existsSync(modelDir)) {
            progress.status = 'failed';
            progress.message = `Model ${modelId} already exists at ${modelDir}`;
            progress.endTime = Date.now();
            return { success: false, message: progress.message };
        }
        return await new Promise((resolve) => {
            const child = spawn('huggingface-cli', ['download', modelId, '--local-dir', modelDir, '--local-dir-use-symlinks', 'False'], { stdio: ['pipe', 'pipe', 'pipe'] });
            progress.childProcess = child;
            let lastOutput = '';
            let totalFiles = 0;
            let downloadedFiles = 0;
            child.stdout.on('data', (data) => {
                const output = data.toString();
                lastOutput = output;
                const fileMatch = output.match(/(\d+)\/(\d+)/);
                if (fileMatch) {
                    downloadedFiles = parseInt(fileMatch[1]);
                    totalFiles = parseInt(fileMatch[2]);
                    progress.progress = totalFiles > 0 ? Math.round((downloadedFiles / totalFiles) * 100) : 0;
                    progress.message = `Downloading ${modelId}... (${downloadedFiles}/${totalFiles} files)`;
                }
                else if (output.includes('Downloading')) {
                    progress.message = `Downloading ${modelId}...`;
                }
                else if (output.includes('Fetching')) {
                    progress.message = `Fetching ${modelId} metadata...`;
                }
            });
            child.stderr.on('data', (data) => {
                const errorOutput = data.toString();
                if (errorOutput.includes('Downloading') || errorOutput.includes('Fetching')) {
                    progress.message = errorOutput.trim();
                }
            });
            child.on('close', (code) => {
                progress.endTime = Date.now();
                if (code === 0) {
                    progress.status = 'completed';
                    progress.progress = 100;
                    progress.message = `Successfully downloaded ${modelId}`;
                    progress.path = modelDir;
                    resolve({ success: true, message: progress.message, path: modelDir });
                }
                else {
                    progress.status = 'failed';
                    progress.error = `Download failed with exit code ${code}`;
                    progress.message = `Failed to download ${modelId}: ${lastOutput || 'Unknown error'}`;
                    resolve({ success: false, message: progress.message });
                }
            });
            child.on('error', (error) => {
                progress.status = 'failed';
                progress.error = error.message;
                progress.message = `Error downloading ${modelId}: ${error.message}`;
                progress.endTime = Date.now();
                resolve({ success: false, message: progress.message });
            });
        });
    }
    catch (error) {
        progress.status = 'failed';
        progress.error = error instanceof Error ? error.message : String(error);
        progress.message = `Error downloading model ${modelId}: ${progress.error}`;
        progress.endTime = Date.now();
        return { success: false, message: progress.message };
    }
}
export function cancelDownload(modelId) {
    const progress = downloadProgress.get(modelId);
    if (!progress)
        return { success: false, message: `No active download found for model ${modelId}` };
    if (progress.status === 'completed' || progress.status === 'failed') {
        return { success: false, message: `Download for model ${modelId} is already ${progress.status}` };
    }
    if (progress.childProcess) {
        progress.childProcess.kill('SIGTERM');
    }
    progress.status = 'cancelled';
    progress.message = `Download cancelled for ${modelId}`;
    progress.endTime = Date.now();
    return { success: true, message: `Download cancelled for model ${modelId}` };
}
