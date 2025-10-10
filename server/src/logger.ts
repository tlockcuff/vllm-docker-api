type LogLevel = 'info' | 'warn' | 'error' | 'debug';

interface LogFields {
  [key: string]: unknown;
}

function serializeValue(value: unknown): string {
  if (value === null || value === undefined) return String(value);
  if (typeof value === 'string') {
    // Quote strings with spaces for readability
    return /\s/.test(value) ? `"${value}"` : value;
  }
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function format(level: LogLevel, message: string, fields?: LogFields) {
  const lvl = level.toUpperCase();
  let suffix = '';
  if (fields && Object.keys(fields).length > 0) {
    const parts: string[] = [];
    for (const [key, value] of Object.entries(fields)) {
      parts.push(`${key}=${serializeValue(value)}`);
    }
    suffix = ' ' + parts.join(' ');
  }
  return `${lvl} ${message}${suffix}`;
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


