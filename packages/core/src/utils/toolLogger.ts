import pino from 'pino';
import path from 'path';
import fs from 'fs';

const logDir = path.resolve(process.cwd(), 'LOGS');
const logFile = path.resolve(logDir, 'TOOLLOG.log');

// Asegurarse de que el directorio de logs exista
if (!fs.existsSync(logDir)) {
  fs.mkdirSync(logDir, { recursive: true });
}

// Configuración para pino-pretty (desarrollo)
const transport = process.env.NODE_ENV !== 'production' ?
  pino.transport({
    targets: [
      {
        target: 'pino-pretty',
        options: {
          colorize: true,
          translateTime: 'SYS:yyyy-mm-dd HH:MM:ss,l',
          ignore: 'pid,hostname',
          destination: logFile,
          mkdir: true, // Crear el archivo de log si no existe
        },
      },
      // También podríamos agregar un target para la consola si fuera necesario
      // {
      //   target: 'pino-pretty',
      //   options: {
      //     colorize: true,
      //     translateTime: 'SYS:yyyy-mm-dd HH:MM:ss,l',
      //     ignore: 'pid,hostname',
      //   }
      // }
    ]
  }) :
  pino.destination(logFile); // En producción, escribir directamente al archivo

const toolLogger = pino(
  {
    level: process.env.LOG_LEVEL || 'info', // Nivel de log configurable
    timestamp: pino.stdTimeFunctions.isoTime, // Usar formato ISO8601 para el timestamp
    formatters: {
      level: (label) => {
        return { level: label.toUpperCase() };
      },
    },
  },
  transport
);

// Manejo de errores del logger
if (transport instanceof require('stream')) {
  transport.on('error', (err) => {
    console.error('Error en el transporte del logger (toolLogger):', err);
  });
} else if (transport && typeof transport.on === 'function') { // Para pino.destination
   transport.on('error', (err: Error) => {
     console.error('Error en el stream del logger (toolLogger):', err);
   });
}


export default toolLogger;
