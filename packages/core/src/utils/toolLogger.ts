// Attempting to use require for pino as a workaround for ESM import issues
// This is not ideal but aims to unblock compilation.
// eslint-disable-next-line @typescript-eslint/no-var-requires
const pino = require('pino');
import path from 'path';
import fs from 'fs';
import { Logger, LoggerOptions, DestinationStream, TransportTargetOptions } from 'pino';


const logDir = path.resolve(process.cwd(), 'LOGS');
const logFile = path.resolve(logDir, 'TOOLLOG.log');

// Ensure the LOGS directory exists
if (!fs.existsSync(logDir)) {
  try {
    fs.mkdirSync(logDir, { recursive: true });
  } catch (mkdirErr) {
    // Log to console if file logger setup fails critically
    console.error(`[toolLogger] Failed to create log directory ${logDir}:`, mkdirErr);
  }
}

const loggerOptions: LoggerOptions = {
  level: process.env.LOG_LEVEL || 'info',
  timestamp: pino.stdTimeFunctions.isoTime,
  formatters: {
    level: (label: string) => ({ level: label.toUpperCase() }),
  },
};

let destination: DestinationStream | ReturnType<typeof pino.transport>;

if (process.env.NODE_ENV !== 'production') {
  const transportOptions: TransportTargetOptions = {
    target: 'pino-pretty',
    options: {
      colorize: true,
      translateTime: 'SYS:yyyy-mm-dd HH:MM:ss,l',
      ignore: 'pid,hostname',
      destination: logFile,
      mkdir: true,
      append: true,
    },
    level: loggerOptions.level,
  };
  destination = pino.transport({ targets: [transportOptions] });
} else {
  destination = pino.destination({
    dest: logFile,
    mkdir: true,
    append: true,
    sync: false, // Recommended for production for better performance
  });
}

const toolLogger: Logger = pino(loggerOptions, destination);

// Graceful shutdown handler
// pino.final should be called with the logger instance
const finalHandler = pino.final(toolLogger, (err: Error | null, finalLogger: Logger, evt: string) => {
  finalLogger.info(`ToolLogger is shutting down due to ${evt}.`);
  if (err) {
    finalLogger.error({ err }, 'Error during shutdown:');
    process.exitCode = 1;
  }
});

process.on('beforeExit', () => finalHandler(null, 'beforeExit'));
process.on('exit', () => finalHandler(null, 'exit'));
process.on('SIGINT', () => { finalHandler(null, 'SIGINT'); process.exit(0); }); // Graceful exit on Ctrl+C
process.on('SIGTERM', () => { finalHandler(null, 'SIGTERM'); process.exit(0); }); // Graceful exit on termination
process.on('uncaughtException', (err) => {
  finalHandler(err, 'uncaughtException');
  process.exit(1); // Exit with error code
});
process.on('unhandledRejection', (reason) => {
  const err = reason instanceof Error ? reason : new Error(String(reason));
  finalHandler(err, 'unhandledRejection');
  process.exit(1); // Exit with error code
});

export default toolLogger;
