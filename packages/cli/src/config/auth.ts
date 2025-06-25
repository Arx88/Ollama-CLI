/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { AuthType } from '@google/gemini-cli-core';
import { loadEnvironment } from './config.js';

export const validateAuthMethod = (authMethod: string): string | null => {
  loadEnvironment();
  if (authMethod === AuthType.LOGIN_WITH_GOOGLE_PERSONAL) {
    return null;
  }

  if (authMethod === AuthType.LOGIN_WITH_GOOGLE_ENTERPRISE) {
    if (!process.env.GOOGLE_CLOUD_PROJECT) {
      return 'GOOGLE_CLOUD_PROJECT environment variable not found. Add that to your .env and try again, no reload needed!';
    }
    return null;
  }

  if (authMethod === AuthType.USE_GEMINI) {
    if (!process.env.GEMINI_API_KEY) {
      return 'GEMINI_API_KEY environment variable not found. Add that to your .env and try again, no reload needed!';
    }
    return null;
  }

  if (authMethod === AuthType.USE_VERTEX_AI) {
    const hasVertexProjectLocationConfig =
      !!process.env.GOOGLE_CLOUD_PROJECT && !!process.env.GOOGLE_CLOUD_LOCATION;
    const hasGoogleApiKey = !!process.env.GOOGLE_API_KEY;
    if (!hasVertexProjectLocationConfig && !hasGoogleApiKey) {
      return (
        'Must specify GOOGLE_GENAI_USE_VERTEXAI=true and either:\n' +
        '• GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION environment variables.\n' +
        '• GOOGLE_API_KEY environment variable (if using express mode).\n' +
        'Update your .env and try again, no reload needed!'
      );
    }
    return null;
  }

  if (authMethod === AuthType.USE_PORTKEY) {
    if (!process.env.PORTKEY_API_KEY) {
      return (
        'PORTKEY_API_KEY environment variable not found.\n' +
        'Also need GEMINI_API_KEY (or PORTKEY_VERTEX_ACCESS_TOKEN) for Vertex AI access.\n' +
        'Optional: PORTKEY_VERTEX_PROJECT_ID, PORTKEY_VERTEX_REGION, PORTKEY_BASE_URL.\n' +
        'Add these to your .env and try again, no reload needed!'
      );
    }
    if (!process.env.GEMINI_API_KEY && !process.env.PORTKEY_VERTEX_ACCESS_TOKEN) {
      return (
        'GEMINI_API_KEY or PORTKEY_VERTEX_ACCESS_TOKEN environment variable not found.\n' +
        'Portkey needs a Gemini API key to access Vertex AI.\n' +
        'Add this to your .env and try again, no reload needed!'
      );
    }
    return null;
  }

  return 'Invalid auth method selected.';
};
