import * as fs from 'fs';
import * as path from 'path';

const logDirectory = path.resolve(process.cwd(), 'LOG');
const logFilePath = path.resolve(logDirectory, 'cli_debug.log');

function ensureLogDirectoryExists(): void {
  if (!fs.existsSync(logDirectory)) {
    fs.mkdirSync(logDirectory, { recursive: true });
  }
}

export function logToFile(message: string): void {
  ensureLogDirectoryExists();
  const timestamp = new Date().toISOString();
  const logMessage = `${timestamp} - ${message}\n`;

  try {
    fs.appendFileSync(logFilePath, logMessage);
  } catch (error) {
    // Fallback to console if file logging fails for some reason
    console.error(`Failed to write to log file: ${error}`);
    console.log(logMessage);
  }
}
