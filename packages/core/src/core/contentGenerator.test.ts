/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi } from 'vitest';
import { createContentGenerator, AuthType } from './contentGenerator.js';
import { createCodeAssistContentGenerator } from '../code_assist/codeAssist.js';
import { GoogleGenAI } from '@google/genai';

vi.mock('../code_assist/codeAssist.js');
vi.mock('@google/genai');

describe('contentGenerator', () => {
  it('should create a CodeAssistContentGenerator', async () => {
    const mockGenerator = {} as unknown;
    vi.mocked(createCodeAssistContentGenerator).mockResolvedValue(
      mockGenerator as never,
    );
    // Mock the Config class that createContentGenerator might use internally
    const mockConfigInstance = {
      getOllamaModel: vi.fn().mockReturnValue('default-ollama-model'),
      // Add other methods from Config that might be called by createContentGenerator
    };
    vi.mock('../config/config.js', () => ({
      Config: vi.fn(() => mockConfigInstance)
    }));

    const generator = await createContentGenerator(
      {
        model: 'test-model',
        authType: AuthType.LOGIN_WITH_GOOGLE_PERSONAL,
      },
      mockConfigInstance as any // Pass the mocked config instance
    );
    expect(createCodeAssistContentGenerator).toHaveBeenCalled();
    expect(generator).toBe(mockGenerator);
  });

  it('should create a GoogleGenAI content generator', async () => {
    const mockGenerator = {
      models: {},
    } as unknown;
    vi.mocked(GoogleGenAI).mockImplementation(() => mockGenerator as never);
    // Mock the Config class that createContentGenerator might use internally
    const mockConfigInstance = {
      getOllamaModel: vi.fn().mockReturnValue('default-ollama-model'),
      // Add other methods from Config that might be called by createContentGenerator
    };
    // No need to re-mock Config if it's already mocked at the top level or per test suite
    // Ensure the mock is effective for this test case too.

    const generator = await createContentGenerator(
      {
        model: 'test-model',
        apiKey: 'test-api-key',
        authType: AuthType.USE_GEMINI,
      },
      mockConfigInstance as any // Pass the mocked config instance
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
