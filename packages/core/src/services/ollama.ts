/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Config } from '../config/config.js';
import { fetchWithRetry } from '../utils/fetch.js'; // Assuming a fetch utility

// TODO: Define these more robustly, perhaps move to a types file
interface OllamaListResponse {
  models: Array<{
    name: string;
    modified_at: string;
    size: number;
    digest: string;
    details: {
      parent_model: string;
      format: string;
      family: string;
      families: string[] | null;
      parameter_size: string;
      quantization_level: string;
    };
  }>;
}

interface OllamaShowModelResponse {
  // Define based on Ollama API documentation for show model details
  // This might include license, modelfile content, parameters, template etc.
  modelfile?: string;
  parameters?: string;
  template?: string;
  details?: object;
  license?: string;
}

export class OllamaClient {
  private readonly config: Config;
  private readonly baseUrl: string;

  constructor(config: Config, baseUrl = 'http://localhost:11434') {
    this.config = config;
    this.baseUrl = baseUrl; // Make this configurable
  }

  private async request<T>(
    endpoint: string,
    method: 'GET' | 'POST' = 'GET',
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    body?: any,
  ): Promise<T> {
    const url = `${this.baseUrl}/api/${endpoint}`;
    const options: RequestInit = {
      method,
      headers: {
        'Content-Type': 'application/json',
      },
    };

    if (body) {
      options.body = JSON.stringify(body);
    }

    if (this.config.getDebugMode()) {
      console.debug(`Ollama Request: ${method} ${url}`, body ? JSON.stringify(body, null, 2) : '');
    }

    const response = await fetchWithRetry(url, options);

    if (!response.ok) {
      const errorBody = await response.text();
      console.error(`Ollama API Error (${response.status} ${response.statusText}): ${errorBody}`);
      throw new Error(
        `Ollama API request failed: ${response.status} ${response.statusText} - ${errorBody}`,
      );
    }

    const responseJson = await response.json();
    if (this.config.getDebugMode()) {
      console.debug('Ollama Response:', JSON.stringify(responseJson, null, 2));
    }
    return responseJson as T;
  }

  async listModels(): Promise<OllamaListResponse> {
    return this.request<OllamaListResponse>('tags');
  }

  async showModelDetails(modelName: string): Promise<OllamaShowModelResponse> {
    return this.request<OllamaShowModelResponse>('show', 'POST', { name: modelName });
  }

  // Placeholder for generate - this will be more complex
  // async generate(modelName: string, prompt: string, stream = false) {
  //   return this.request('generate', 'POST', {
  //     model: modelName,
  //     prompt: prompt,
  //     stream: stream,
  //   });
  // }

  async generate(
    params: OllamaGenerateParams,
  ): Promise<OllamaGenerateResponse | AsyncGenerator<OllamaGenerateResponse>> {
    const url = `${this.baseUrl}/api/generate`;
    const options: RequestInit = {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(params),
    };

    if (this.config.getDebugMode()) {
      console.debug(`Ollama Generate Request: POST ${url}`, JSON.stringify(params, null, 2));
    }

    const response = await fetchWithRetry(url, options);

    if (!response.ok) {
      const errorBody = await response.text();
      console.error(`Ollama API Error (${response.status} ${response.statusText}): ${errorBody}`);
      throw new Error(
        `Ollama API request failed: ${response.status} ${response.statusText} - ${errorBody}`,
      );
    }

    if (params.stream === false) {
      const responseJson = await response.json();
      if (this.config.getDebugMode()) {
        console.debug('Ollama Generate Response (non-streaming):', JSON.stringify(responseJson, null, 2));
      }
      return responseJson as OllamaGenerateResponse;
    } else {
      // Handle streaming response
      const client = this; // For use inside the generator function
      async function* streamGenerator(): AsyncGenerator<OllamaGenerateResponse> {
        if (!response.body) {
          throw new Error('Response body is null for streaming request.');
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            if (buffer.trim()) { // Process any remaining data in buffer
              try {
                const jsonChunk = JSON.parse(buffer.trim());
                 if (client.config.getDebugMode()) {
                    console.debug('Ollama Generate Response (streaming chunk):', JSON.stringify(jsonChunk, null, 2));
                  }
                yield jsonChunk as OllamaGenerateResponse;
              } catch (e) {
                console.error('Error parsing final JSON chunk in stream:', e, buffer);
              }
            }
            break;
          }
          buffer += decoder.decode(value, { stream: true });
          let newlineIndex;
          while ((newlineIndex = buffer.indexOf('\n')) !== -1) {
            const line = buffer.slice(0, newlineIndex).trim();
            buffer = buffer.slice(newlineIndex + 1);
            if (line) {
              try {
                const jsonChunk = JSON.parse(line);
                if (client.config.getDebugMode()) {
                    console.debug('Ollama Generate Response (streaming chunk):', JSON.stringify(jsonChunk, null, 2));
                }
                yield jsonChunk as OllamaGenerateResponse;
              } catch (e) {
                console.error('Error parsing JSON chunk in stream:', e, line);
                // Optionally, decide if you want to throw or continue
              }
            }
          }
        }
      }
      return streamGenerator();
    }
  }

  async embeddings(
    params: OllamaEmbeddingsParams,
  ): Promise<OllamaEmbeddingsResponse> {
    // The 'request' method can be reused if the error handling and logging are similar.
    // Otherwise, a more specific implementation like in 'generate' might be needed.
    return this.request<OllamaEmbeddingsResponse>('embeddings', 'POST', params);
  }
}

// Type definitions for Ollama Generate endpoint
// Based on https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion
export interface OllamaGenerateParams {
  model: string;
  prompt: string;
  system?: string;
  template?: string;
  context?: number[]; // Context from previous generations
  stream?: boolean; // Default true, but we'll control it
  raw?: boolean; // Use raw prompt, default false
  format?: 'json'; // If format is json, Ollama will attempt to output valid JSON
  images?: string[]; // base64 encoded images (for multimodal models)
  options?: Record<string, unknown>; // Model parameters, e.g., temperature, top_p
}

export interface OllamaGenerateResponse {
  model: string;
  created_at: string;
  response: string; // The generated text for this chunk if streaming
  done: boolean; // True if this is the final response
  done_reason?: string; // e.g. "stop", "length"

  // If not streaming and not raw:
  context?: number[]; // Context for next generation
  total_duration?: number;
  load_duration?: number;
  prompt_eval_count?: number;
  prompt_eval_duration?: number;
  eval_count?: number;
  eval_duration?: number;
}

// Type definitions for Ollama Embeddings endpoint
// Based on https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
export interface OllamaEmbeddingsParams {
  model: string;
  prompt: string;
  options?: Record<string, unknown>; // Model parameters
}

export interface OllamaEmbeddingsResponse {
  embedding: number[];
}
