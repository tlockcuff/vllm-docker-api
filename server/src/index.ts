import { createApp } from './app.js';
import { PORT } from './config.js';
import { logger } from './logger.js';

const app = createApp();
app.listen(PORT, () => {
  logger.info('server_start', { url: `http://localhost:${PORT}`, port: PORT });
});


