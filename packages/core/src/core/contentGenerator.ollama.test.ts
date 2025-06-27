/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  createContentGenerator,
  AuthType,
  ContentGeneratorConfig,
  OllamaContentGenerator,
} from './contentGenerator.js';
import {
  OllamaClient,
  OllamaGenerateResponse,
  OllamaEmbeddingsResponse,
} from '../services/ollama.js';
import { Config } from '../config/config.js';
import {
  GenerateContentParameters,
  EmbedContentParameters,
  FinishReason,
  CountTokensParameters,
} from '@google/genai';
import toolLogger from '../utils/toolLogger.js'; // Import toolLogger at the top

// Mock toolLogger first as other mocks might depend on it or its side effects
vi.mock('../utils/toolLogger.js', () => ({
  default: {
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
    debug: vi.fn(),
    trace: vi.fn(),
    child: vi.fn().mockReturnThis(),
  },
}));

// Mock OllamaClient
vi.mock('../services/ollama.js', async () => {
  const actual = await vi.importActual('../services/ollama.js');
  return {
    ...actual,
    OllamaClient: vi.fn().mockImplementation(() => ({
      generate: vi.fn(),
      embeddings: vi.fn(),
      listModels: vi.fn(),
    })),
  };
});

// Mock Config
vi.mock('../config/config.js', () => ({
  Config: vi.fn().mockImplementation(() => ({
    getDebugMode: vi.fn().mockReturnValue(false),
    getOllamaModel: vi.fn().mockReturnValue('test-ollama-model'),
  })),
}));

describe('OllamaContentGenerator', () => {
  let mockOllamaClient: OllamaClient;
  let ollamaGenerator: OllamaContentGenerator;
  let mockConfig: Config;
  let toolLoggerWarnSpy: vi.MockInstance;

  beforeEach(() => {
    // Reset mocks for Config and OllamaClient to ensure clean state for each test
    // This is important if createContentGenerator is called in multiple tests
    vi.mocked(Config).mockClear();
    vi.mocked(OllamaClient).mockClear();

    const mockConfigParams = {
      sessionId: 'test-session',
      targetDir: '/test',
      debugMode: false,
      model: 'gemini-pro',
      ollamaModel: 'test-model-from-params', // Distinct from getOllamaModel default
      cwd: '/test',
    };

    mockConfig = new Config(mockConfigParams);
    // Ensure getOllamaModel on this specific instance returns what's needed for generator creation
    vi.mocked(mockConfig.getOllamaModel).mockReturnValue('test-model-from-config-get');


    // Create a new mock client for each test to reset call counts etc.
    // Note: OllamaClient is already mocked globally. We are creating an instance of the mocked client.
    mockOllamaClient = new OllamaClient(mockConfig);

    ollamaGenerator = new OllamaContentGenerator(
      'test-model-instance', // Specific model for this instance
      mockOllamaClient,
    );

    // Spy on toolLogger.warn before each test
    toolLoggerWarnSpy = vi.spyOn(toolLogger, 'warn');
  });

  afterEach(() => {
    vi.clearAllMocks(); // Clears all mocks, including spies
  });

  describe('generateContent', () => {
    it('should transform request, call ollamaClient.generate, and transform response', async () => {
      const geminiRequest: GenerateContentParameters = {
        contents: [{ role: 'user', parts: [{ text: 'Hello Ollama' }] }],
      };
      const ollamaApiResponse: OllamaGenerateResponse = {
        model: 'test-model-instance',
        created_at: 'timestamp',
        response: 'Ollama says hello',
        done: true,
        done_reason: 'stop',
        context: [1, 2, 3],
        eval_count: 10,
        prompt_eval_count: 5,
      };
      vi.mocked(mockOllamaClient.generate).mockResolvedValue(ollamaApiResponse);

      const result = await ollamaGenerator.generateContent(geminiRequest);

      expect(mockOllamaClient.generate).toHaveBeenCalledWith(
        expect.objectContaining({
          model: 'test-model-instance',
          prompt: 'Hello Ollama',
          stream: false,
        }),
      );
      const candidate = result.candidates?.[0];
      expect(candidate?.content?.parts?.[0]?.text).toBe('Ollama says hello');
      expect(candidate?.finishReason).toBe(FinishReason.STOP);
      expect(candidate?.tokenCount).toBe(10);
    });
  });

  describe('generateContentStream', () => {
    it('should handle streaming response from ollamaClient.generate', async () => {
      const geminiRequest: GenerateContentParameters = {
        contents: [{ role: 'user', parts: [{ text: 'Stream test' }] }],
      };
      const mockStreamChunks: OllamaGenerateResponse[] = [
        { model: 'test-model-instance', created_at: 't1', response: 'Chunk 1 ', done: false },
        { model: 'test-model-instance', created_at: 't2', response: 'Chunk 2', done: true, done_reason: 'stop', context: [4, 5, 6], eval_count: 5 },
      ];

      async function* mockAsyncGenerator() {
        for (const chunk of mockStreamChunks) {
          yield chunk;
        }
      }
      vi.mocked(mockOllamaClient.generate).mockResolvedValue(mockAsyncGenerator());

      const stream = await ollamaGenerator.generateContentStream(geminiRequest);
      const receivedResponses = [];
      for await (const response of stream) {
        receivedResponses.push(response);
      }

      expect(mockOllamaClient.generate).toHaveBeenCalledWith(
        expect.objectContaining({
          model: 'test-model-instance',
          prompt: 'Stream test',
          stream: true,
        }),
      );
      expect(receivedResponses.length).toBe(2);
      expect(receivedResponses[0].candidates?.[0]?.content?.parts?.[0]?.text).toBe('Chunk 1 ');
      expect(receivedResponses[1].candidates?.[0]?.content?.parts?.[0]?.text).toBe('Chunk 2');
      expect(receivedResponses[1].candidates?.[0]?.finishReason).toBe(FinishReason.STOP);
    });
  });

  describe('embedContent', () => {
    it('should call ollamaClient.embeddings and return formatted response', async () => {
      const geminiRequest: EmbedContentParameters = {
        contents: { role: 'user', parts: [{ text: 'Embed this text' }] },
      };
      const ollamaApiResponse: OllamaEmbeddingsResponse = {
        embedding: [0.1, 0.2, 0.3, 0.4],
      };
      vi.mocked(mockOllamaClient.embeddings).mockResolvedValue(ollamaApiResponse);

      const result = await ollamaGenerator.embedContent(geminiRequest);

      expect(mockOllamaClient.embeddings).toHaveBeenCalledWith(
        expect.objectContaining({
          model: 'test-model-instance',
          prompt: 'Embed this text',
        }),
      );
      expect(result.embeddings?.[0]?.values).toEqual([0.1, 0.2, 0.3, 0.4]);
    });

    it('should handle string content for embeddings', async () => {
      const geminiRequest: EmbedContentParameters = {
        contents: 'Embed this string directly',
      };
      const ollamaApiResponse: OllamaEmbeddingsResponse = { embedding: [0.5, 0.6] };
      vi.mocked(mockOllamaClient.embeddings).mockResolvedValue(ollamaApiResponse);
      const result = await ollamaGenerator.embedContent(geminiRequest);
      expect(mockOllamaClient.embeddings).toHaveBeenCalledWith(
        expect.objectContaining({ prompt: 'Embed this string directly' }),
      );
      expect(result.embeddings?.[0]?.values).toEqual([0.5, 0.6]);
    });

    it('should throw if prompt text is empty for embeddings', async () => {
      const geminiRequest: EmbedContentParameters = { contents: { parts: [] } };
      await expect(ollamaGenerator.embedContent(geminiRequest)).rejects.toThrow(
        'Prompt text is required for Ollama embedContent.',
      );
    });
  });

  describe('countTokens', () => {
    it('should return estimated token count and log warning', async () => {
      const geminiRequest: CountTokensParameters = {
        contents: [{ role: 'user', parts: [{ text: 'Tokenize this for me please' }] }],
      };
      const expectedEstimatedTokens = Math.ceil('Tokenize this for me please'.length / 3.5);
      const result = await ollamaGenerator.countTokens(geminiRequest);

      expect(result.totalTokens).toBe(expectedEstimatedTokens);
      expect(toolLogger.warn).toHaveBeenCalledWith(
        expect.stringContaining('Ollama countTokens is using a very rough character-based estimation'),
        expect.objectContaining({ model: 'test-model-instance' })
      );
    });
  });

  describe('createContentGenerator for Ollama', () => {
    it('should create an OllamaContentGenerator instance when authType is USE_OLLAMA', async () => {
      const contentGeneratorConfig: ContentGeneratorConfig = {
        model: 'ollama-model-from-cfg', // This is the model name that should be used by the created generator
        authType: AuthType.USE_OLLAMA,
      };

      // Create a new mockConfig specifically for this test, if needed, or use the global one
      // and ensure its getOllamaModel is properly mocked for this call path.
      const specificMockConfig = new Config({} as any); // Basic mock
      vi.mocked(specificMockConfig.getOllamaModel).mockReturnValue('ollama-model-from-cfg');


      const generator = await createContentGenerator(contentGeneratorConfig, specificMockConfig);

      expect(generator).toBeInstanceOf(OllamaContentGenerator);
      // To check the model name used by the created generator, we'd ideally access a property.
      // Since modelName is private, we can check if OllamaClient was constructed with the correct config
      // (which implies the model name was passed correctly to OllamaContentGenerator constructor).
      // This requires OllamaClient mock to be more sophisticated or to inspect its constructor calls.
      // For now, this test primarily ensures the type of generator.
      // We also need to ensure that the OllamaClient constructor is called within createContentGenerator.
      // The global mock `vi.mocked(OllamaClient)` tracks all calls to its constructor.
      expect(OllamaClient).toHaveBeenCalled();
    });
  });
});
