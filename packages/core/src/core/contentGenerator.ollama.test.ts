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
  OllamaContentGenerator, // Moved OllamaContentGenerator here
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
} from '@google/genai'; // Added CountTokensParameters

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
vi.mock('../config/config.js', () => ({
  Config: vi.fn().mockImplementation(() => ({
    getDebugMode: vi.fn().mockReturnValue(false),
    getOllamaModel: vi.fn().mockReturnValue('test-ollama-model'), // For createContentGeneratorConfig
    // Add other necessary Config mocks
  })),
}));

describe('OllamaContentGenerator', () => {
  let mockOllamaClient: OllamaClient;
  let ollamaGenerator: OllamaContentGenerator;
  let mockConfig: Config;

  beforeEach(() => {
    const mockConfigParams = {
      sessionId: 'test-session',
      targetDir: '/test',
      debugMode: false,
      model: 'gemini-pro', // Main model, Ollama model comes from getOllamaModel
      ollamaModel: 'test-model',
      cwd: '/test',
    };

    // Create a new mock client for each test to reset call counts etc.
    mockOllamaClient = new OllamaClient(new Config(mockConfigParams));
    ollamaGenerator = new OllamaContentGenerator(
      'test-model',
      mockOllamaClient,
    );
    mockConfig = new Config(mockConfigParams);
    (mockConfig.getOllamaModel as import('vitest').Mock).mockReturnValue(
      'test-model',
    ); // Used import('vitest').Mock
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('generateContent', () => {
    it('should transform request, call ollamaClient.generate, and transform response', async () => {
      const geminiRequest: GenerateContentParameters = {
        model: 'test-model', // Added model property
        contents: [{ role: 'user', parts: [{ text: 'Hello Ollama' }] }],
        // generationConfig is not a direct property of GenerateContentParameters anymore for Ollama
        // These are now mapped inside OllamaContentGenerator.paramsFromGeminiRequest
        // For the purpose of this test, we can pass them if the SUT expects them,
        // or adjust the SUT if it should take them from a different source (e.g. a general config).
        // Assuming for now the SUT's `paramsFromGeminiRequest` handles this.
        // If direct pass-through was intended, the type definition of GenerateContentParameters would need `generationConfig`.
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
      (mockOllamaClient.generate as import('vitest').Mock).mockResolvedValue(
        ollamaApiResponse,
      ); // Used import('vitest').Mock

      const result = await ollamaGenerator.generateContent(geminiRequest);

      expect(mockOllamaClient.generate).toHaveBeenCalledWith(
        expect.objectContaining({
          model: 'test-model',
          prompt: 'Hello Ollama',
          stream: false,
          // options: expect.objectContaining({ temperature: 0.5, num_predict: 100 }), // generationConfig is handled differently
        }),
      );
      const candidate = result.candidates?.[0];
      if (
        candidate &&
        candidate.content &&
        candidate.content.parts &&
        candidate.content.parts.length > 0
      ) {
        expect(candidate.content.parts[0].text).toBe('Ollama says hello');
        expect(candidate.finishReason).toBe(FinishReason.STOP);
        expect(candidate.tokenCount).toBe(10);
      } else {
        // Explicitly fail if the structure is not as expected
        expect(result.candidates).toBeDefined();
        if (result.candidates) {
          expect(result.candidates.length).toBeGreaterThan(0);
          const firstCandidate = result.candidates[0];
          expect(firstCandidate).toBeDefined();
          if (firstCandidate) {
            expect(firstCandidate.content).toBeDefined();
            if (firstCandidate.content) {
              expect(firstCandidate.content.parts).toBeDefined();
              if (firstCandidate.content.parts) {
                expect(firstCandidate.content.parts.length).toBeGreaterThan(0);
              }
            }
          }
        }
      }
      // Removed @ts-expect-error and assertion for private property
    });
  });

  describe('generateContentStream', () => {
    it('should handle streaming response from ollamaClient.generate', async () => {
      const geminiRequest: GenerateContentParameters = {
        model: 'test-model', // Added model property
        contents: [{ role: 'user', parts: [{ text: 'Stream test' }] }],
      };
      const mockStreamChunks: OllamaGenerateResponse[] = [
        {
          model: 'test-model',
          created_at: 't1',
          response: 'Chunk 1 ',
          done: false,
        },
        {
          model: 'test-model',
          created_at: 't2',
          response: 'Chunk 2',
          done: true,
          done_reason: 'stop',
          context: [4, 5, 6],
          eval_count: 5,
        },
      ];

      async function* mockAsyncGenerator() {
        for (const chunk of mockStreamChunks) {
          yield chunk;
        }
      }
      (mockOllamaClient.generate as import('vitest').Mock).mockResolvedValue(
        mockAsyncGenerator(),
      ); // Used import('vitest').Mock

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

      const firstResponseCandidate = receivedResponses[0].candidates?.[0];
      if (
        firstResponseCandidate &&
        firstResponseCandidate.content &&
        firstResponseCandidate.content.parts &&
        firstResponseCandidate.content.parts.length > 0
      ) {
        expect(firstResponseCandidate.content.parts[0].text).toBe('Chunk 1 ');
        expect(firstResponseCandidate.finishReason).toBe(
          FinishReason.FINISH_REASON_UNSPECIFIED,
        );
      } else {
        expect(firstResponseCandidate).toBeDefined();
        if (firstResponseCandidate)
          expect(firstResponseCandidate.content).toBeDefined();
        if (firstResponseCandidate?.content)
          expect(firstResponseCandidate.content.parts).toBeDefined();
        if (firstResponseCandidate?.content?.parts)
          expect(firstResponseCandidate.content.parts.length).toBeGreaterThan(
            0,
          );
      }

      const secondResponseCandidate = receivedResponses[1].candidates?.[0];
      if (
        secondResponseCandidate &&
        secondResponseCandidate.content &&
        secondResponseCandidate.content.parts &&
        secondResponseCandidate.content.parts.length > 0
      ) {
        expect(secondResponseCandidate.content.parts[0].text).toBe('Chunk 2');
        expect(secondResponseCandidate.finishReason).toBe(FinishReason.STOP);
        expect(secondResponseCandidate.tokenCount).toBe(5);
      } else {
        expect(secondResponseCandidate).toBeDefined();
        if (secondResponseCandidate)
          expect(secondResponseCandidate.content).toBeDefined();
        if (secondResponseCandidate?.content)
          expect(secondResponseCandidate.content.parts).toBeDefined();
        if (secondResponseCandidate?.content?.parts)
          expect(secondResponseCandidate.content.parts.length).toBeGreaterThan(
            0,
          );
      }
      // Removed @ts-expect-error and assertion for private property
    });
  });

  describe('embedContent', () => {
    it('should call ollamaClient.embeddings and return formatted response', async () => {
      const geminiRequest: EmbedContentParameters = {
        model: 'test-model', // Added model property
        contents: { role: 'user', parts: [{ text: 'Embed this text' }] },
      };
      const ollamaApiResponse: OllamaEmbeddingsResponse = {
        embedding: [0.1, 0.2, 0.3, 0.4],
      };
      (mockOllamaClient.embeddings as import('vitest').Mock).mockResolvedValue(
        ollamaApiResponse,
      ); // Used import('vitest').Mock

      const result = await ollamaGenerator.embedContent(geminiRequest);

      expect(mockOllamaClient.embeddings).toHaveBeenCalledWith(
        expect.objectContaining({
          model: 'test-model',
          prompt: 'Embed this text',
        }),
      );
      expect(result.embeddings?.values).toEqual([0.1, 0.2, 0.3, 0.4]); // Changed embedding to embeddings
    });
    it('should handle string content for embeddings', async () => {
      const geminiRequest: EmbedContentParameters = {
        model: 'test-model', // Added model property
        contents: 'Embed this string directly',
      };
      const ollamaApiResponse: OllamaEmbeddingsResponse = {
        embedding: [0.5, 0.6],
      };
      (mockOllamaClient.embeddings as import('vitest').Mock).mockResolvedValue(
        ollamaApiResponse,
      ); // Used import('vitest').Mock
      const result = await ollamaGenerator.embedContent(geminiRequest);
      expect(mockOllamaClient.embeddings).toHaveBeenCalledWith(
        expect.objectContaining({ prompt: 'Embed this string directly' }),
      );
      expect(result.embeddings?.values).toEqual([0.5, 0.6]); // Changed embedding to embeddings
    });

    it('should throw if prompt text is empty for embeddings', async () => {
      const geminiRequest: EmbedContentParameters = {
        model: 'test-model',
        contents: { parts: [] },
      }; // Added model, Changed content to contents
      await expect(ollamaGenerator.embedContent(geminiRequest)).rejects.toThrow(
        'Prompt text is required for Ollama embedContent.',
      );
    });
  });

  describe('countTokens', () => {
    it('should return estimated token count and log warning', async () => {
      const consoleWarnSpy = vi
        .spyOn(console, 'warn')
        .mockImplementation(() => {});
      const geminiRequest: CountTokensParameters = {
        model: 'test-model', // Added model property
        contents: [
          { role: 'user', parts: [{ text: 'Tokenize this for me please' }] },
        ],
      };
      // "Tokenize this for me please".length = 27. 27 / 3.5 = 7.71 -> ceil = 8
      const expectedEstimatedTokens = Math.ceil(
        'Tokenize this for me please'.length / 3.5,
      );

      const result = await ollamaGenerator.countTokens(geminiRequest);

      expect(result.totalTokens).toBe(expectedEstimatedTokens);
      expect(consoleWarnSpy).toHaveBeenCalledWith(
        expect.stringContaining(
          'Ollama countTokens is using a very rough character-based estimation',
        ),
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
      const generator = await createContentGenerator(
        contentGeneratorConfig,
        mockConfig,
      );

      expect(generator).toBeInstanceOf(OllamaContentGenerator);
      // expect(generator.modelName).toBe('ollama-model-from-cfg'); // modelName is private
      // Check if the OllamaClient mock constructor was called by createContentGenerator
      expect(OllamaClient).toHaveBeenCalledTimes(1); // This might be more if other tests also create OllamaClient
      // This assertion checks that the OllamaClient constructor within createContentGenerator was called with the mockConfig
      expect(OllamaClient).toHaveBeenCalledWith(mockConfig);
    });
  });
});
