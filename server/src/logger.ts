type LogLevel = 'info' | 'warn' | 'error' | 'debug';

interface LogFields {
  [key: string]: unknown;
}

function format(level: LogLevel, message: string, fields?: LogFields) {
  const base = {
    level,
    message: message,
  } as Record<string, unknown>;
  const payload = fields ? { ...base, ...fields } : base;
  return JSON.stringify(payload);
}

export const logger = {
  info(message: string, fields?: LogFields) {
    console.log(format('info', message, fields));
  },
  warn(message: string, fields?: LogFields) {
    console.warn(format('warn', message, fields));
  },
  error(message: string, fields?: LogFields) {
    console.error(format('error', message, fields));
  },
  debug(message: string, fields?: LogFields) {
    if (process.env.NODE_ENV !== 'production') {
      console.debug(format('debug', message, fields));
    }
  },
};

export type { LogFields };


