import fs from 'fs';
import path from 'path';
import util from 'util';

const logDir = path.resolve(process.cwd(), 'LOGS');
const logFile = path.resolve(logDir, 'TOOLLOG.log');

// Ensure the LOGS directory exists
if (!fs.existsSync(logDir)) {
  try {
    fs.mkdirSync(logDir, { recursive: true });
  } catch (mkdirErr: any) {
    console.error(`[SimpleToolLogger] CRITICAL: Failed to create log directory ${logDir}: ${mkdirErr.message}`);
  }
}

const LOG_LEVELS: { [key: string]: number } = {
  trace: 1,
  debug: 2,
  info: 3,
  warn: 4,
  error: 5,
  fatal: 6,
};

const currentLogLevelName = (process.env.TOOL_LOG_LEVEL || 'info').toLowerCase();
const currentLogLevel = LOG_LEVELS[currentLogLevelName] || LOG_LEVELS.info;

function formatData(data?: Record<string, any> | Error): string {
  if (!data) return '';
  if (data instanceof Error) {
    return `Error: ${data.message}\nStack: ${data.stack}`;
  }
  try {
    // util.inspect might be too verbose for large objects, but good for circular refs
    return util.inspect(data, { depth: 4, colors: false }); // depth can be adjusted
  } catch (e) {
    return 'Could not stringify data';
  }
}

function writeLog(level: string, message: string, data?: Record<string, any> | Error) {
  const timestamp = new Date().toISOString();
  const dataString = formatData(data);
  const logMessage = `${timestamp} - ${level.toUpperCase()} - ${message}${dataString ? ` - Data: ${dataString}` : ''}\n`;

  try {
    fs.appendFileSync(logFile, logMessage, { encoding: 'utf8' });
  } catch (appendErr: any) {
    console.error(`[SimpleToolLogger] CRITICAL: Failed to append to log file ${logFile}: ${appendErr.message}`);
  }
}

const toolLogger = {
  trace: (message: string, data?: Record<string, any> | Error) => {
    if (currentLogLevel <= LOG_LEVELS.trace) writeLog('trace', message, data);
  },
  debug: (message: string, data?: Record<string, any> | Error) => {
    if (currentLogLevel <= LOG_LEVELS.debug) writeLog('debug', message, data);
  },
  info: (message: string, data?: Record<string, any> | Error) => {
    if (currentLogLevel <= LOG_LEVELS.info) writeLog('info', message, data);
  },
  warn: (message: string, data?: Record<string, any> | Error) => {
    if (currentLogLevel <= LOG_LEVELS.warn) writeLog('warn', message, data);
  },
  error: (message: string, data?: Record<string, any> | Error) => {
    if (currentLogLevel <= LOG_LEVELS.error) writeLog('error', message, data);
  },
  fatal: (message: string, data?: Record<string, any> | Error) => {
    // 'fatal' is often an alias for error in simple loggers or implies process exit
    if (currentLogLevel <= LOG_LEVELS.fatal) writeLog('fatal', message, data);
  },
  // Add a child-like method for compatibility with pino's API if needed, though it won't have true child logger features.
  child: (bindings: Record<string, any>) => {
    // This simple logger doesn't have true child loggers.
    // We can return a new logger instance that includes bindings in its messages.
    // Or, more simply, return itself and ignore bindings for now.
    // For now, let's make it so it prefixes messages or includes bindings in data.
    const childLogger = {
        trace: (message: string, data?: Record<string, any> | Error) => toolLogger.trace(message, {...bindings, ...data}),
        debug: (message: string, data?: Record<string, any> | Error) => toolLogger.debug(message, {...bindings, ...data}),
        info: (message: string, data?: Record<string, any> | Error) => toolLogger.info(message, {...bindings, ...data}),
        warn: (message: string, data?: Record<string, any> | Error) => toolLogger.warn(message, {...bindings, ...data}),
        error: (message: string, data?: Record<string, any> | Error) => toolLogger.error(message, {...bindings, ...data}),
        fatal: (message: string, data?: Record<string, any> | Error) => toolLogger.fatal(message, {...bindings, ...data}),
        child: (furtherBindings: Record<string, any>) => toolLogger.child({...bindings, ...furtherBindings}), // Allow chaining child
    };
    // Add an info log to show a child logger was created, including its bindings
    toolLogger.info('Child logger created', bindings);
    return childLogger;
  }
};

// Log initialization
toolLogger.info(`SimpleToolLogger initialized. Log level set to: ${currentLogLevelName.toUpperCase()} (${currentLogLevel})`);
toolLogger.info(`Logs will be written to: ${logFile}`);

export default toolLogger;
