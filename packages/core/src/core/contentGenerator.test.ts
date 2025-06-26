/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, vi, expect } from 'vitest';
import { createContentGenerator, AuthType } from './contentGenerator.js';
import { GoogleGenAI } from '@google/genai';
import { Config } from '../config/config.js';

vi.mock('../code_assist/codeAssist.js');
vi.mock('@google/genai');

describe('contentGenerator', () => {
  it('should create a CodeAssistContentGenerator', async () => {
    const mockConfigInstance: Partial<Config> = {
      getOllamaModel: vi.fn().mockReturnValue('default-ollama-model'),
      getDebugMode: vi.fn().mockReturnValue(false),
      getOllamaApiEndpoint: vi.fn().mockReturnValue('http://localhost:11434'),
    };
    vi.mock('../config/config.js', () => ({
      Config: vi.fn(() => mockConfigInstance as Config),
    }));

    await createContentGenerator(
      {
        model: 'test-model',
        authType: AuthType.LOGIN_WITH_GOOGLE_PERSONAL,
      },
      mockConfigInstance as Config,
    );
  });

  it('should create a GoogleGenAI content generator', async () => {
    const mockGenerator = {
      models: {},
    } as unknown;
    vi.mocked(GoogleGenAI).mockImplementation(() => mockGenerator as never);
    // Mock the Config class that createContentGenerator might use internally
    const mockConfigInstance: Partial<Config> = {
      getOllamaModel: vi.fn().mockReturnValue('default-ollama-model'),
      getDebugMode: vi.fn().mockReturnValue(false),
      getOllamaApiEndpoint: vi.fn().mockReturnValue('http://localhost:11434'),
    };
    // No need to re-mock Config if it's already mocked at the top level or per test suite
    // Ensure the mock is effective for this test case too.

    const generator = await createContentGenerator(
      {
        model: 'test-model',
        apiKey: 'test-api-key',
        authType: AuthType.USE_GEMINI,
      },
      mockConfigInstance as Config,
    );
    expect(GoogleGenAI).toHaveBeenCalledWith({
      apiKey: 'test-api-key',
      vertexai: undefined,
      httpOptions: {
        headers: {
          'User-Agent': expect.any(String),
        },
      },
    });
    expect(generator).toBe((mockGenerator as GoogleGenAI).models);
  });
});
