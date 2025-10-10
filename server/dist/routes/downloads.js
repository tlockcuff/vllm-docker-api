import { registry } from '../openapi.js';
import { DownloadModelRequestSchema } from '../schemas.js';
import { cancelDownload, downloadHuggingFaceModel, downloadProgress } from '../services/downloads.js';
export function mountDownloadRoutes(app) {
    registry.registerPath({ method: 'post', path: '/api/models/download', request: { body: { content: { 'application/json': { schema: DownloadModelRequestSchema } } } }, responses: { 200: { description: 'Started or finished download' } } });
    registry.registerPath({ method: 'get', path: '/api/models/download/progress', responses: { 200: { description: 'Progress for all or specific model' } } });
    registry.registerPath({ method: 'get', path: '/api/models/download/progress/{model}', responses: { 200: { description: 'Progress for model' } } });
    registry.registerPath({ method: 'delete', path: '/api/models/download/{model}', responses: { 200: { description: 'Cancel download' } } });
    app.post('/api/models/download', async (req, res) => {
        try {
            const requestData = DownloadModelRequestSchema.parse(req.body);
            const result = await downloadHuggingFaceModel(requestData.model);
            if (result.success) {
                res.json({ success: true, message: result.message, model: requestData.model, path: result.path });
            }
            else {
                res.status(400).json({ success: false, message: result.message, model: requestData.model });
            }
        }
        catch (e) {
            if (e.issues) {
                res.status(400).json({ success: false, message: 'Invalid request data', details: e.issues });
            }
            else {
                res.status(500).json({ success: false, message: String(e) });
            }
        }
    });
    app.get('/api/models/download/progress', async (req, res) => {
        try {
            const modelParam = req.query.model;
            if (modelParam) {
                const progress = downloadProgress.get(modelParam);
                if (!progress) {
                    return res.status(404).json({ success: false, message: `No download progress found for model ${modelParam}` });
                }
                const { childProcess, ...rest } = progress;
                return res.json({ success: true, progress: { ...rest, duration: progress.endTime ? progress.endTime - progress.startTime : Date.now() - progress.startTime } });
            }
            else {
                const allProgress = Array.from(downloadProgress.entries()).map(([_, { childProcess, ...rest }]) => ({ ...rest, duration: rest.endTime ? rest.endTime - rest.startTime : Date.now() - rest.startTime }));
                return res.json({ success: true, downloads: allProgress });
            }
        }
        catch (error) {
            res.status(500).json({ success: false, message: `Error retrieving download progress: ${error instanceof Error ? error.message : String(error)}` });
        }
    });
    app.get('/api/models/download/progress/:model', async (req, res) => {
        try {
            const model = req.params.model;
            const progress = downloadProgress.get(model);
            if (!progress) {
                return res.status(404).json({ success: false, message: `No download progress found for model ${model}` });
            }
            const { childProcess, ...rest } = progress;
            res.json({ success: true, progress: { ...rest, duration: progress.endTime ? progress.endTime - progress.startTime : Date.now() - progress.startTime } });
        }
        catch (error) {
            res.status(500).json({ success: false, message: `Error retrieving download progress: ${error instanceof Error ? error.message : String(error)}` });
        }
    });
    app.delete('/api/models/download/:model', async (req, res) => {
        try {
            const model = req.params.model;
            const result = cancelDownload(model);
            if (result.success) {
                res.json(result);
            }
            else {
                const code = /already/.test(result.message) ? 400 : 404;
                res.status(code).json(result);
            }
        }
        catch (error) {
            res.status(500).json({ success: false, message: `Error cancelling download: ${error instanceof Error ? error.message : String(error)}` });
        }
    });
}
