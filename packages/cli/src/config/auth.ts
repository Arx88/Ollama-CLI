/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { AuthType } from '@google/gemini-cli-core';
import { loadEnvironment } from './config.js';
import { logToFile } from '../utils/fileLogger.js';

export const validateAuthMethod = (authMethod: string): string | null => {
  logToFile(`[validateAuthMethod] Received authMethod: ${authMethod}`);
  loadEnvironment();
  if (authMethod === AuthType.LOGIN_WITH_GOOGLE_PERSONAL) {
    logToFile(
      '[validateAuthMethod] Returning null for LOGIN_WITH_GOOGLE_PERSONAL',
    );
    return null;
  }

  if (authMethod === AuthType.USE_GEMINI) {
    if (!process.env.GEMINI_API_KEY) {
      const errorMsg =
        'GEMINI_API_KEY environment variable not found. Add that to your .env and try again, no reload needed!';
      logToFile(
        `[validateAuthMethod] Returning error for USE_GEMINI: ${errorMsg}`,
      );
      return errorMsg;
    }
    logToFile('[validateAuthMethod] Returning null for USE_GEMINI');
    return null;
  }

  if (authMethod === AuthType.USE_VERTEX_AI) {
    const hasVertexProjectLocationConfig =
      !!process.env.GOOGLE_CLOUD_PROJECT && !!process.env.GOOGLE_CLOUD_LOCATION;
    const hasGoogleApiKey = !!process.env.GOOGLE_API_KEY;
    if (!hasVertexProjectLocationConfig && !hasGoogleApiKey) {
      const errorMsg =
        'Must specify GOOGLE_GENAI_USE_VERTEXAI=true and either:\n' +
        '• GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION environment variables.\n' +
        '• GOOGLE_API_KEY environment variable (if using express mode).\n' +
        'Update your .env and try again, no reload needed!';
      logToFile(
        `[validateAuthMethod] Returning error for USE_VERTEX_AI: ${errorMsg}`,
      );
      return errorMsg;
    }
    logToFile('[validateAuthMethod] Returning null for USE_VERTEX_AI');
    return null;
  }

  if (authMethod === AuthType.USE_OLLAMA) {
    logToFile('[validateAuthMethod] Returning null for USE_OLLAMA');
    return null; // Ollama runs locally, no specific auth validation needed here.
  }

  logToFile('[validateAuthMethod] Returning "Invalid auth method selected."');
  return 'Invalid auth method selected.';
};
