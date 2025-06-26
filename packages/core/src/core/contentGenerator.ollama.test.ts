/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  OllamaContentGenerator,
  createContentGenerator,
  AuthType,
  ContentGeneratorConfig,
} from './contentGenerator.js';
import { OllamaClient, OllamaGenerateResponse, OllamaEmbeddingsResponse } from '../services/ollama.js';
import { Config } from '../config/config.js';
import { GenerateContentParameters, EmbedContentParameters, FinishReason, HarmCategory, HarmBlockThreshold } from '@google/genai';

// Mock OllamaClient
vi.mock('../services/ollama.js', async () => {
  const actual = await vi.importActual('../services/ollama.js');
  return {
    ...actual,
    OllamaClient: vi.fn().mockImplementation(() => ({
      generate: vi.fn(),
      embeddings: vi.fn(),
      listModels: vi.fn(), // Add other methods if they get called during generator setup
    })),
  };
});

// Mock Config
vi.mock('../config/config.js', () => {
  return {
    Config: vi.fn().mockImplementation(() => ({
      getDebugMode: vi.fn().mockReturnValue(false),
      getOllamaModel: vi.fn().mockReturnValue('test-ollama-model'), // For createContentGeneratorConfig
      // Add other necessary Config mocks
    })),
  };
});


describe('OllamaContentGenerator', () => {
  let mockOllamaClient: OllamaClient;
  let ollamaGenerator: OllamaContentGenerator;
  let mockConfig: Config;

  beforeEach(() => {
    // Create a new mock client for each test to reset call counts etc.
    mockOllamaClient = new OllamaClient(new Config({} as any) /* mock config if needed by OllamaClient directly */);
    ollamaGenerator = new OllamaContentGenerator('test-model', mockOllamaClient);
    mockConfig = new Config({
        sessionId: 'test-session',
        targetDir: '/test',
        debugMode: false,
        model: 'gemini-pro', // Main model, Ollama model comes from getOllamaModel
        ollamaModel: 'test-model',
        cwd: '/test',
    });
    (mockConfig.getOllamaModel as vi.Mock).mockReturnValue('test-model');

  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('generateContent', () => {
    it('should transform request, call ollamaClient.generate, and transform response', async () => {
      const geminiRequest: GenerateContentParameters = {
        contents: [{ role: 'user', parts: [{ text: 'Hello Ollama' }] }],
        generationConfig: { temperature: 0.5, maxOutputTokens: 100 },
      };
      const ollamaApiResponse: OllamaGenerateResponse = {
        model: 'test-model',
        created_at: 'timestamp',
        response: 'Ollama says hello',
        done: true,
        done_reason: 'stop',
        context: [1, 2, 3],
        eval_count: 10,
        prompt_eval_count: 5,
      };
      (mockOllamaClient.generate as vi.Mock).mockResolvedValue(ollamaApiResponse);

      const result = await ollamaGenerator.generateContent(geminiRequest);

      expect(mockOllamaClient.generate).toHaveBeenCalledWith(
        expect.objectContaining({
          model: 'test-model',
          prompt: 'Hello Ollama',
          stream: false,
          options: expect.objectContaining({ temperature: 0.5, num_predict: 100 }),
        }),
      );
      expect(result.candidates[0].content.parts[0].text).toBe('Ollama says hello');
      expect(result.candidates[0].finishReason).toBe(FinishReason.STOP);
      expect(result.candidates[0].tokenCount).toBe(10);
      // @ts-expect-error - currentContext is private
      expect(ollamaGenerator.currentContext).toEqual([1, 2, 3]);
    });
  });

  describe('generateContentStream', () => {
    it('should handle streaming response from ollamaClient.generate', async () => {
      const geminiRequest: GenerateContentParameters = {
        contents: [{ role: 'user', parts: [{ text: 'Stream test' }] }],
      };
      const mockStreamChunks: OllamaGenerateResponse[] = [
        { model: 'test-model', created_at: 't1', response: 'Chunk 1 ', done: false },
        { model: 'test-model', created_at: 't2', response: 'Chunk 2', done: true, done_reason: 'stop', context: [4,5,6], eval_count: 5 },
      ];

      async function* mockAsyncGenerator() {
        for (const chunk of mockStreamChunks) {
          yield chunk;
        }
      }
      (mockOllamaClient.generate as vi.Mock).mockResolvedValue(mockAsyncGenerator());

      const stream = await ollamaGenerator.generateContentStream(geminiRequest);
      const receivedResponses = [];
      for await (const response of stream) {
        receivedResponses.push(response);
      }

      expect(mockOllamaClient.generate).toHaveBeenCalledWith(
        expect.objectContaining({
          model: 'test-model',
          prompt: 'Stream test',
          stream: true,
        }),
      );
      expect(receivedResponses.length).toBe(2);
      expect(receivedResponses[0].candidates[0].content.parts[0].text).toBe('Chunk 1 ');
      expect(receivedResponses[0].candidates[0].finishReason).toBe(FinishReason.FINISH_REASON_UNSPECIFIED);
      expect(receivedResponses[1].candidates[0].content.parts[0].text).toBe('Chunk 2');
      expect(receivedResponses[1].candidates[0].finishReason).toBe(FinishReason.STOP);
      expect(receivedResponses[1].candidates[0].tokenCount).toBe(5);
      // @ts-expect-error - currentContext is private
      expect(ollamaGenerator.currentContext).toEqual([4,5,6]);
    });
  });

  describe('embedContent', () => {
    it('should call ollamaClient.embeddings and return formatted response', async () => {
      const geminiRequest: EmbedContentParameters = {
        content: { role: 'user', parts: [{text: 'Embed this text'}] },
      };
      const ollamaApiResponse: OllamaEmbeddingsResponse = {
        embedding: [0.1, 0.2, 0.3, 0.4],
      };
      (mockOllamaClient.embeddings as vi.Mock).mockResolvedValue(ollamaApiResponse);

      const result = await ollamaGenerator.embedContent(geminiRequest);

      expect(mockOllamaClient.embeddings).toHaveBeenCalledWith(
        expect.objectContaining({
          model: 'test-model',
          prompt: 'Embed this text',
        }),
      );
      expect(result.embedding.values).toEqual([0.1, 0.2, 0.3, 0.4]);
    });
     it('should handle string content for embeddings', async () => {
      const geminiRequest: EmbedContentParameters = {
        content: 'Embed this string directly',
      };
      const ollamaApiResponse: OllamaEmbeddingsResponse = {
        embedding: [0.5, 0.6],
      };
      (mockOllamaClient.embeddings as vi.Mock).mockResolvedValue(ollamaApiResponse);
      const result = await ollamaGenerator.embedContent(geminiRequest);
      expect(mockOllamaClient.embeddings).toHaveBeenCalledWith(
        expect.objectContaining({ prompt: 'Embed this string directly' }),
      );
      expect(result.embedding.values).toEqual([0.5, 0.6]);
    });

    it('should throw if prompt text is empty for embeddings', async () => {
      const geminiRequest: EmbedContentParameters = { content: { parts: [] } };
      await expect(ollamaGenerator.embedContent(geminiRequest)).rejects.toThrow(
        'Prompt text is required for Ollama embedContent.'
      );
    });
  });

  describe('countTokens', () => {
    it('should return estimated token count and log warning', async () => {
      const consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
      const geminiRequest: CountTokensParameters = {
        contents: [{ role: 'user', parts: [{ text: 'Tokenize this for me please' }] }],
      };
      // "Tokenize this for me please".length = 27. 27 / 3.5 = 7.71 -> ceil = 8
      const expectedEstimatedTokens = Math.ceil("Tokenize this for me please".length / 3.5);

      const result = await ollamaGenerator.countTokens(geminiRequest);

      expect(result.totalTokens).toBe(expectedEstimatedTokens);
      expect(consoleWarnSpy).toHaveBeenCalledWith(
        expect.stringContaining('Ollama countTokens is using a very rough character-based estimation'),
      );
      consoleWarnSpy.mockRestore();
    });
  });

  describe('createContentGenerator for Ollama', () => {
    it('should create an OllamaContentGenerator instance when authType is USE_OLLAMA', async () => {
      const contentGeneratorConfig: ContentGeneratorConfig = {
        model: 'ollama-model-from-cfg', // This will be used as effectiveModel
        authType: AuthType.USE_OLLAMA,
      };

      // Ensure the global mock for OllamaClient is used when createContentGenerator calls `new OllamaClient()`
      const generator = await createContentGenerator(contentGeneratorConfig, mockConfig);

      expect(generator).toBeInstanceOf(OllamaContentGenerator);
      // @ts-expect-error - modelName is private but we want to check it
      expect(generator.modelName).toBe('ollama-model-from-cfg');
      // Check if the OllamaClient mock constructor was called by createContentGenerator
      expect(OllamaClient).toHaveBeenCalledTimes(1);
      // This assertion checks that the OllamaClient constructor within createContentGenerator was called with the mockConfig
      expect(OllamaClient).toHaveBeenCalledWith(mockConfig);
    });
  });
});
