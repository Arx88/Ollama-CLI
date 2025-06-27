/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Config } from '../config/config.js';
import { fetchWithTimeout } from '../utils/fetch.js';
import toolLogger from '../utils/toolLogger.js';
import { randomUUID } from 'crypto';

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
      console.debug(
        `Ollama Request: ${method} ${url}`,
        body ? JSON.stringify(body, null, 2) : '',
      );
    }

    // Default timeout for Ollama requests, e.g., 30 seconds for non-streaming
    // Streaming requests might need longer or different handling if fetchWithTimeout isn't ideal for streams.
    // For now, using a relatively long timeout for all requests made via this.request
    // The `generate` method uses fetchWithTimeout directly for more control over streaming.
    const response = await fetchWithTimeout(url, 60000, options); // 60 second timeout

    if (!response.ok) {
      const errorBody = await response.text();
      // Removed toolLogger.error from here as requestId and params are not in scope
      console.error(
        `Ollama API Error in request() (${response.status} ${response.statusText}): ${errorBody}`,
      );
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
    return this.request<OllamaShowModelResponse>('show', 'POST', {
      name: modelName,
    });
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
    const requestId = randomUUID();
    toolLogger.info('Ollama generate request initiated', {
      requestId,
      action: 'ollama_generate_start',
      model: params.model,
      stream: params.stream,
      tool_count: params.tools?.length || 0,
    });
    if (params.tools && params.tools.length > 0) {
      toolLogger.debug('Ollama generate tool definitions', { requestId, tools: params.tools });
    }

    const url = `${this.baseUrl}/api/generate`;
    const options: RequestInit = {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(params),
    };

    if (this.config.getDebugMode()) {
      console.debug(
        `Ollama Generate Request: POST ${url}`,
        JSON.stringify(params, null, 2),
      );
    }

    // For streaming, a very long timeout or no timeout via fetchWithTimeout might be preferable,
    // as the stream duration is unknown. fetchWithTimeout will abort if the *total* request time
    // (including all stream chunks) exceeds the timeout.
    // If params.stream is true, we might consider using raw fetch or a very long timeout.
    // For non-streaming, a reasonable timeout is good.
    const timeout = params.stream ? 300000 : 60000; // 5 mins for stream, 1 min for non-stream
    const response = await fetchWithTimeout(url, timeout, options);

    if (!response.ok) {
      const errorBody = await response.text();
      // Log error with the simple logger: toolLogger.error(message, data)
      toolLogger.error(`Ollama API Error: ${response.status} ${response.statusText}`, {
        requestId,
        action: 'ollama_generate_error',
        status: response.status,
        statusText: response.statusText,
        errorBody,
        model: params.model,
      });
      throw new Error(
        `Ollama API request failed: ${response.status} ${response.statusText} - ${errorBody}`,
      );
    }

    if (params.stream === false) {
      const responseJson = (await response.json()) as OllamaGenerateResponse;
      toolLogger.info('Ollama generate non-streaming response received.', {
        requestId,
        action: 'ollama_generate_non_stream_response',
        model: params.model,
        done: responseJson.done,
        tool_calls_count: responseJson.tool_calls?.length || 0,
      });
      if (responseJson.tool_calls && responseJson.tool_calls.length > 0) {
        toolLogger.debug('Non-streaming tool_calls content', { requestId, tool_calls: responseJson.tool_calls });
      }
      // if (this.config.getDebugMode()) { // Original debug logging
      //   console.debug(
      //     'Ollama Generate Response (non-streaming):',
      //     JSON.stringify(responseJson, null, 2),
      //   );
      // }
      return responseJson;
    } else {
      // Handle streaming response
      toolLogger.info('Ollama generate streaming response started.', {
        requestId,
        action: 'ollama_generate_stream_start',
        model: params.model,
      });
      // Pass requestId and model for logging within the generator
      async function* streamGenerator(currentRequestId: string, currentModel: string): AsyncGenerator<OllamaGenerateResponse> {
        if (!response.body) {
          toolLogger.error('Response body is null for streaming request.', {
            currentRequestId,
            action: 'ollama_stream_error',
            model: currentModel,
          });
          throw new Error('Response body is null for streaming request.');
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            if (buffer.trim()) {
              // Process any remaining data in buffer
              try {
                const jsonChunk = JSON.parse(buffer.trim()) as OllamaGenerateResponse;
                toolLogger.debug('Ollama generate final stream chunk from buffer processed.', {
                  currentRequestId,
                  action: 'ollama_stream_chunk',
                  model: currentModel,
                  done: jsonChunk.done,
                  final_chunk_buffer: true,
                  tool_calls_count: jsonChunk.tool_calls?.length || 0,
                });
                if (jsonChunk.tool_calls && jsonChunk.tool_calls.length > 0) {
                  toolLogger.debug('Final stream chunk (from buffer) tool_calls content', {
                    currentRequestId,
                    tool_calls: jsonChunk.tool_calls,
                  });
                }
                yield jsonChunk;
              } catch (e: any) {
                toolLogger.error('Error parsing final JSON chunk in stream', {
                  currentRequestId,
                  action: 'ollama_stream_parse_error',
                  model: currentModel,
                  error: e.message,
                  buffer,
                });
              }
            }
            toolLogger.info('Ollama generate stream ended.', {
              currentRequestId,
              action: 'ollama_generate_stream_end',
              model: currentModel,
            });
            break;
          }
          buffer += decoder.decode(value, { stream: true });
          let newlineIndex;
          while ((newlineIndex = buffer.indexOf('\n')) !== -1) {
            const line = buffer.slice(0, newlineIndex).trim();
            buffer = buffer.slice(newlineIndex + 1);
            if (line) {
              try {
                const jsonChunk = JSON.parse(line) as OllamaGenerateResponse;
                toolLogger.debug('Ollama generate stream chunk processed.', {
                  currentRequestId,
                  action: 'ollama_stream_chunk',
                  model: currentModel,
                  done: jsonChunk.done,
                  tool_calls_count: jsonChunk.tool_calls?.length || 0,
                });
                if (jsonChunk.tool_calls && jsonChunk.tool_calls.length > 0) {
                  toolLogger.debug('Stream chunk tool_calls content', {
                    currentRequestId,
                    tool_calls: jsonChunk.tool_calls,
                  });
                }
                yield jsonChunk;
              } catch (e: any) {
                toolLogger.error('Error parsing JSON chunk in stream', {
                  currentRequestId,
                  action: 'ollama_stream_parse_error',
                  model: currentModel,
                  error: e.message,
                  line,
                });
              }
            }
          }
        }
      }
      return streamGenerator(requestId, params.model); // Pass requestId and model
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
  tools?: Array<{
    type: 'function';
    function: {
      name: string;
      description?: string;
      parameters: Record<string, unknown>;
    };
  }>;
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

  // Added to support tool calling
  tool_calls?: Array<{
    function: {
      name: string;
      parameters: Record<string, unknown>;
    };
  }>;
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
