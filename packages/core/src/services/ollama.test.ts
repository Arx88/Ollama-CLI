/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi, beforeEach, afterEach, Mock } from 'vitest'; // Changed MockInstance to Mock
import { OllamaClient, OllamaGenerateResponse } from './ollama.js';
import { Config } from '../config/config.js';

// Define a simple fetch mock utility directly in the test file
const mockFetch = {
  responses: new Map<string, { body: string; status: number }>(),
  streamingResponses: new Map<string, string>(),
  fetch: vi.fn() as Mock<
    (url: RequestInfo | URL, _options?: RequestInit) => Promise<Response>
  >,
  setup: () => {
    global.fetch = mockFetch.fetch;
    mockFetch.fetch.mockImplementation(
      async (url: RequestInfo | URL, _options?: RequestInit) => {
        const urlString = url.toString();
        if (mockFetch.streamingResponses.has(urlString)) {
          const streamData = mockFetch.streamingResponses.get(urlString)!;
          const readableStream = new ReadableStream({
            start(controller) {
              const lines = streamData.split('\n');
              lines.forEach((line) =>
                controller.enqueue(new TextEncoder().encode(line + '\n')),
              );
              controller.close();
            },
          });
          return Promise.resolve(new Response(readableStream, { status: 200 }));
        }
        if (mockFetch.responses.has(urlString)) {
          const { body, status } = mockFetch.responses.get(urlString)!;
          return Promise.resolve(new Response(body, { status }));
        }
        return Promise.resolve(new Response('Not Found', { status: 404 }));
      },
    );
  },
  reset: () => {
    mockFetch.responses.clear();
    mockFetch.streamingResponses.clear();
    mockFetch.fetch.mockClear();
  },
  addResponse: (url: string, body: string, status = 200) => {
    mockFetch.responses.set(url, { body, status });
  },
  addStreamingResponse: (url: string, body: string) => {
    mockFetch.streamingResponses.set(url, body);
  },
};

// Mock the Config class
vi.mock('../config/config.js', () => ({
  Config: vi.fn().mockImplementation(() => ({
    getDebugMode: vi.fn().mockReturnValue(false),
    // Add other methods that might be called by OllamaClient if necessary
  })),
}));

describe('OllamaClient', () => {
  let config: Config;
  let ollamaClient: OllamaClient;

  beforeEach(() => {
    const mockConfigParams = {
      sessionId: 'test-session',
      targetDir: '/test',
      debugMode: false,
      model: 'gemini-pro',
      cwd: '/test',
    };
    config = new Config(mockConfigParams);
    ollamaClient = new OllamaClient(config, 'http://localhost:11434');
    mockFetch.setup();
  });

  afterEach(() => {
    mockFetch.reset();
    vi.clearAllMocks();
  });

  describe('listModels', () => {
    it('should fetch and return a list of models', async () => {
      const mockModels = {
        models: [
          {
            name: 'llama2:latest',
            modified_at: '2023-10-26T14:00:00Z',
            size: 12345,
          },
          {
            name: 'mistral:latest',
            modified_at: '2023-10-27T15:00:00Z',
            size: 67890,
          },
        ],
      };
      mockFetch.addResponse(
        'http://localhost:11434/api/tags',
        JSON.stringify(mockModels),
      );

      const models = await ollamaClient.listModels();
      expect(models).toEqual(mockModels);
      expect(mockFetch.fetch).toHaveBeenCalledTimes(1);
      expect(mockFetch.fetch).toHaveBeenCalledWith(
        'http://localhost:11434/api/tags',
        expect.objectContaining({ method: 'GET' }),
      );
    });

    it('should throw an error if the API request fails', async () => {
      mockFetch.addResponse(
        'http://localhost:11434/api/tags',
        'Internal Server Error',
        500,
      );

      await expect(ollamaClient.listModels()).rejects.toThrow(
        'Ollama API request failed: 500  - Internal Server Error',
      );
    });
  });

  describe('showModelDetails', () => {
    it('should fetch and return details for a specific model', async () => {
      const modelName = 'llama2:latest';
      const mockModelDetails = {
        modelfile: '# Modelfile for Llama2',
        parameters: 'num_ctx: 4096',
      };
      mockFetch.addResponse(
        'http://localhost:11434/api/show',
        JSON.stringify(mockModelDetails),
      );

      const details = await ollamaClient.showModelDetails(modelName);
      expect(details).toEqual(mockModelDetails);
      expect(mockFetch.fetch).toHaveBeenCalledTimes(1);
      expect(mockFetch.fetch).toHaveBeenCalledWith(
        'http://localhost:11434/api/show',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ name: modelName }),
        }),
      );
    });

    it('should throw an error if showing model details fails', async () => {
      mockFetch.addResponse(
        'http://localhost:11434/api/show',
        'Model not found',
        404,
      );

      await expect(
        ollamaClient.showModelDetails('nonexistent-model'),
      ).rejects.toThrow(
         'Ollama API request failed: 404  - Model not found',
      );
    });
  });

  describe('Debug Mode Logging', () => {
    it('should log request and response when debug mode is enabled', async () => {
      const consoleDebugSpy = vi
        .spyOn(console, 'debug')
        .mockImplementation(() => {});
      // Create a new config instance with debug mode true
      const debugConfig: Config = new Config({
        sessionId: 'test-session-debug',
        targetDir: '/test',
        debugMode: true, // Enable debug mode
        model: 'gemini-pro',
        cwd: '/test',
      });
      (debugConfig.getDebugMode as import('vitest').Mock).mockReturnValue(true); // Ensure mock returns true

      const debugClient = new OllamaClient(
        debugConfig,
        'http://localhost:11434',
      );

      const mockModels = { models: [{ name: 'llama2:latest' }] };
      mockFetch.addResponse(
        'http://localhost:11434/api/tags',
        JSON.stringify(mockModels),
      );

      await debugClient.listModels();

      expect(consoleDebugSpy).toHaveBeenCalledWith(
        'Ollama Request: GET http://localhost:11434/api/tags',
        '', // No body for GET
      );
      expect(consoleDebugSpy).toHaveBeenCalledWith(
        'Ollama Response:',
        JSON.stringify(mockModels, null, 2),
      );
      consoleDebugSpy.mockRestore();
    });
  });

  describe('generate', () => {
    const generateParams = { model: 'test-model', prompt: 'Hello' };

    it('should send a non-streaming generate request and return response', async () => {
      const mockResponse = {
        model: 'test-model',
        created_at: '2023-11-20T10:00:00Z',
        response: 'World',
        done: true,
        context: [1, 2, 3],
      };
      mockFetch.addResponse(
        'http://localhost:11434/api/generate',
        JSON.stringify(mockResponse),
      );

      const result = await ollamaClient.generate({
        ...generateParams,
        stream: false,
      });

      expect(mockFetch.fetch).toHaveBeenCalledWith(
        'http://localhost:11434/api/generate',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ ...generateParams, stream: false }),
        }),
      );
      expect(result).toEqual(mockResponse);
    });

    it('should handle streaming generate request', async () => {
      const mockStreamChunks = [
        { model: 'test-model', response: 'Hel', done: false },
        { model: 'test-model', response: 'lo, ', done: false },
        {
          model: 'test-model',
          response: 'World!',
          done: true,
          context: [1, 2, 3],
        },
      ];
      const streamString = mockStreamChunks
        .map((chunk) => JSON.stringify(chunk))
        .join('\n');

      mockFetch.addStreamingResponse(
        'http://localhost:11434/api/generate',
        streamString,
      );

      const stream = (await ollamaClient.generate({
        ...generateParams,
        stream: true,
      })) as AsyncGenerator<OllamaGenerateResponse>;

      const receivedChunks = [];
      for await (const chunk of stream) {
        receivedChunks.push(chunk);
      }

      expect(mockFetch.fetch).toHaveBeenCalledWith(
        'http://localhost:11434/api/generate',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ ...generateParams, stream: true }),
        }),
      );
      expect(receivedChunks).toEqual(mockStreamChunks);
    });
    it('should correctly parse stream with remaining buffer content', async () => {
      const mockStreamChunks = [
        {
          model: 'test-model',
          response: 'Final chunk',
          done: true,
          context: [1, 2, 3],
        },
      ];
      // Simulate a situation where the last chunk doesn't have a trailing newline
      const streamString = JSON.stringify(mockStreamChunks[0]);

      mockFetch.addStreamingResponse(
        'http://localhost:11434/api/generate',
        streamString,
      );

      const stream = (await ollamaClient.generate({
        ...generateParams,
        stream: true,
      })) as AsyncGenerator<OllamaGenerateResponse>;

      const receivedChunks = [];
      for await (const chunk of stream) {
        receivedChunks.push(chunk);
      }

      expect(mockFetch.fetch).toHaveBeenCalledWith(
        'http://localhost:11434/api/generate',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ ...generateParams, stream: true }),
        }),
      );
      expect(receivedChunks).toEqual(mockStreamChunks);
    });
  });

  describe('embeddings', () => {
    const embeddingsParams = { model: 'test-model', prompt: 'Embed this' };
    it('should send an embeddings request and return response', async () => {
      const mockResponse = { embedding: [0.1, 0.2, 0.3] };
      mockFetch.addResponse(
        'http://localhost:11434/api/embeddings',
        JSON.stringify(mockResponse),
      );

      const result = await ollamaClient.embeddings(embeddingsParams);

      expect(mockFetch.fetch).toHaveBeenCalledWith(
        'http://localhost:11434/api/embeddings',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify(embeddingsParams),
        }),
      );
      expect(result).toEqual(mockResponse);
    });

    it('should throw an error if embeddings API request fails', async () => {
      mockFetch.addResponse(
        'http://localhost:11434/api/embeddings',
        'Server error',
        500,
      );
      await expect(ollamaClient.embeddings(embeddingsParams)).rejects.toThrow(
        'Ollama API request failed: 500  - Server error',
      );
    });
  });
});
