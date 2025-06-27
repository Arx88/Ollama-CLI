/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  CountTokensResponse,
  GenerateContentResponse,
  GenerateContentParameters,
  CountTokensParameters,
  EmbedContentResponse,
  EmbedContentParameters,
  GoogleGenAI,
} from '@google/genai';
import {
  FinishReason,
  HarmCategory,
  SafetyRating,
  HarmProbability, // Added HarmProbability import
} from '@google/genai';
import { createCodeAssistContentGenerator } from '../code_assist/codeAssist.js';
import { DEFAULT_GEMINI_MODEL } from '../config/models.js';
import { getEffectiveModel } from './modelCheck.js';
import {
  OllamaClient,
  OllamaEmbeddingsParams,
  OllamaEmbeddingsResponse,
  OllamaGenerateParams,
  OllamaGenerateResponse,
} from '../services/ollama.js'; // Import Ollama types
import toolLogger from '../utils/toolLogger.js'; // Import the logger
import { Tool } from '@google/genai'; // Ensure Tool is imported if used in ExtendedGenerateContentParameters

/**
 * Interface abstracting the core functionalities for generating content and counting tokens.
 */
export interface ContentGenerator {
  generateContent(
    request: GenerateContentParameters,
  ): Promise<GenerateContentResponse>;

  generateContentStream(
    request: GenerateContentParameters,
  ): Promise<AsyncGenerator<GenerateContentResponse>>;

  countTokens(request: CountTokensParameters): Promise<CountTokensResponse>;

  embedContent(request: EmbedContentParameters): Promise<EmbedContentResponse>;
}

export enum AuthType {
  LOGIN_WITH_GOOGLE_PERSONAL = 'oauth-personal',
  USE_GEMINI = 'gemini-api-key',
  USE_VERTEX_AI = 'vertex-ai',
  USE_OLLAMA = 'ollama',
}

export type ContentGeneratorConfig = {
  model: string;
  apiKey?: string;
  vertexai?: boolean;
  authType?: AuthType | undefined;
};

export async function createContentGeneratorConfig(
  model: string | undefined,
  authType: AuthType | undefined,
  config?: Config,
): Promise<ContentGeneratorConfig> {
  const geminiApiKey = process.env.GEMINI_API_KEY;
  const googleApiKey = process.env.GOOGLE_API_KEY;
  const googleCloudProject = process.env.GOOGLE_CLOUD_PROJECT;
  const googleCloudLocation = process.env.GOOGLE_CLOUD_LOCATION;
  // Ollama specific config
  const ollamaModel = config?.getOllamaModel?.() || process.env.OLLAMA_MODEL; // Type assertion

  let effectiveModel: string;
  if (authType === AuthType.USE_OLLAMA) {
    effectiveModel = ollamaModel || 'llama2'; // Fallback to a default ollama model
  } else {
    // Use runtime model from config if available, otherwise fallback to parameter or default
    effectiveModel = config?.getModel?.() || model || DEFAULT_GEMINI_MODEL;
  }

  const contentGeneratorConfig: ContentGeneratorConfig = {
    model: effectiveModel, // This will be Ollama model name if authType is USE_OLLAMA
    authType,
  };

  // if we are using google auth nothing else to validate for now
  if (authType === AuthType.LOGIN_WITH_GOOGLE_PERSONAL) {
    return contentGeneratorConfig;
  }

  //
  if (authType === AuthType.USE_GEMINI && geminiApiKey) {
    contentGeneratorConfig.apiKey = geminiApiKey;
    contentGeneratorConfig.model = await getEffectiveModel(
      contentGeneratorConfig.apiKey,
      contentGeneratorConfig.model,
    );

    return contentGeneratorConfig;
  }

  if (
    authType === AuthType.USE_VERTEX_AI &&
    !!googleApiKey &&
    googleCloudProject &&
    googleCloudLocation
  ) {
    contentGeneratorConfig.apiKey = googleApiKey;
    contentGeneratorConfig.vertexai = true;
    contentGeneratorConfig.model = await getEffectiveModel(
      contentGeneratorConfig.apiKey,
      contentGeneratorConfig.model,
    );

    return contentGeneratorConfig;
  }

  // For Ollama, the model is already set in effectiveModel, no API key or Vertex AI setup needed.
  if (authType === AuthType.USE_OLLAMA) {
    return contentGeneratorConfig;
  }

  return contentGeneratorConfig;
}

// Define an extended interface for GenerateContentParameters to include 'tools'
interface ExtendedGenerateContentParameters extends GenerateContentParameters {
  tools?: Tool[];
}

// Placeholder for Ollama Content Generator
// This would need to implement the ContentGenerator interface
// and interact with the Ollama API (likely via OllamaClient).
export class OllamaContentGenerator implements ContentGenerator {
  // Added export here
  private readonly modelName: string;
  private readonly ollamaClient: OllamaClient;
  private currentContext: number[] | undefined; // To store context for conversational turns

  constructor(modelName: string, ollamaClient: OllamaClient) {
    this.modelName = modelName;
    this.ollamaClient = ollamaClient;
    toolLogger.info(`OllamaContentGenerator initialized with model: ${this.modelName}`, {
      generator: 'OllamaContentGenerator',
      model: this.modelName,
    });
  }

  private paramsFromGeminiRequest(
    request: ExtendedGenerateContentParameters, // Use the extended interface
  ): OllamaGenerateParams {
    const loggerContext = {
      generator: 'OllamaContentGenerator',
      method: 'paramsFromGeminiRequest',
      model: this.modelName,
    };
    toolLogger.debug('Converting Gemini request to Ollama params.', loggerContext);

    // TODO: More sophisticated mapping, especially for 'contents' (multi-turn chat history)
    // For now, concatenate text parts from the last user message.
    let promptText = '';
    if (Array.isArray(request.contents) && request.contents.length > 0) {
      const lastContentItem = request.contents[request.contents.length - 1]; // Type: string | Content
      // Ensure lastContentItem is a Content object (has 'role' and 'parts')
      if (
        typeof lastContentItem !== 'string' &&
        'parts' in lastContentItem &&
        Array.isArray(lastContentItem.parts)
      ) {
        promptText = lastContentItem.parts
          .map((part: import('@google/genai').Part) => part.text || '')
          .join(' ');
      }
    }
    // No direct handling for request.contents being a single Content object here,
    // but other functions (countTokens, embedContent) might need it.
    // This function seems to assume request.contents is always an array for multi-turn.

    // Simplistic mapping for generationConfig - Ollama options are different
    // Simplistic mapping for generationConfig - Ollama options are different
    const options: Record<string, unknown> = {};
    // Access generation parameters via the request.config object
    if (request.config?.temperature !== undefined) {
      options.temperature = request.config.temperature;
    }
    if (request.config?.topP !== undefined) {
      options.top_p = request.config.topP;
    }
    if (request.config?.topK !== undefined) {
      options.top_k = request.config.topK;
    }
    if (request.config?.maxOutputTokens !== undefined) {
      options.num_predict = request.config.maxOutputTokens;
    }
    if (request.config?.stopSequences !== undefined) {
      options.stop = request.config.stopSequences;
    }
    // (e.g. first message with role 'system' or a dedicated field if available)

    // Define the structure of a successfully mapped Ollama tool for type guarding
    type MappedOllamaTool = {
      type: 'function';
      function: {
        name: string;
        description: string;
        parameters: Record<string, unknown>;
      };
    };

    let ollamaToolsFormatted: OllamaGenerateParams['tools'] | undefined = undefined;
    if (request.tools && Array.isArray(request.tools)) {
      ollamaToolsFormatted = request.tools
        .map((tool: Tool): MappedOllamaTool | null => {
          if (tool.functionDeclarations && tool.functionDeclarations.length > 0) {
            const funcDecl = tool.functionDeclarations[0];
            if (funcDecl && funcDecl.name) {
              return {
                type: 'function' as const,
                function: {
                  name: funcDecl.name,
                  description: funcDecl.description || '',
                  parameters: (funcDecl.parameters as Record<string, unknown>) || {},
                },
              };
            }
          }
          return null;
        })
        .filter((t): t is MappedOllamaTool => t !== null);

      if (ollamaToolsFormatted.length > 0) {
        toolLogger.debug('Mapped Gemini tools to Ollama format.', { ...loggerContext, tool_count: ollamaToolsFormatted.length });
        toolLogger.trace('Ollama tool definitions.', { ...loggerContext, ollamaTools: ollamaToolsFormatted });
      } else {
        ollamaToolsFormatted = undefined; // Ensure it's undefined if no tools were successfully mapped
      }
    }

    const params: OllamaGenerateParams = {
      model: this.modelName,
      prompt: promptText,
      system: undefined, // Placeholder for system prompt
      template: undefined, // Placeholder for template
      context: this.currentContext, // Pass context from previous turn
      stream: false, // Will be overridden by caller if needed for streaming
      options: Object.keys(options).length > 0 ? options : undefined,
      tools: ollamaToolsFormatted,
    };
    toolLogger.trace('Ollama params created.', { ...loggerContext, params });
    return params;
  }

  private responseToGeminiResponse(
    ollamaResponse: OllamaGenerateResponse,
    requestId?: string, // Optional requestId for correlating logs
  ): GenerateContentResponse {
    const loggerContext = {
      generator: 'OllamaContentGenerator',
      method: 'responseToGeminiResponse',
      model: this.modelName,
      requestId, // Log if provided
      ollama_done: ollamaResponse.done,
      ollama_done_reason: ollamaResponse.done_reason,
      has_tool_calls: !!(ollamaResponse.tool_calls && ollamaResponse.tool_calls.length > 0),
    };
    toolLogger.debug('Converting Ollama response to Gemini response.', loggerContext);

    // Store context for next turn
    if (ollamaResponse.context) {
      this.currentContext = ollamaResponse.context;
      toolLogger.trace('Updated Ollama context.', { ...loggerContext, context_length: ollamaResponse.context.length });
    }

    let finishReason: FinishReason = FinishReason.FINISH_REASON_UNSPECIFIED;
    if (ollamaResponse.done) {
      switch (ollamaResponse.done_reason) {
        case 'stop':
          finishReason = FinishReason.STOP;
          break;
        case 'length':
          finishReason = FinishReason.MAX_TOKENS;
          break;
        default:
          finishReason = FinishReason.OTHER;
      }
    }

    // Ollama doesn't have the same safety rating system as Gemini.
    // We'll return empty or default safety ratings.
    const safetyRatings: SafetyRating[] = [
      {
        category: HarmCategory.HARM_CATEGORY_UNSPECIFIED,
        probability: HarmProbability.NEGLIGIBLE,
      },
    ];

    let text: string | undefined = undefined;
    let parts: Array<import('@google/genai').Part> = [];
    let topLevelFunctionCalls: Array<import('@google/genai').FunctionCall> | undefined = undefined;

    if (ollamaResponse.tool_calls && ollamaResponse.tool_calls.length > 0) {
      toolLogger.info('Received tool_calls from Ollama.', {
        ...loggerContext,
        tool_call_count: ollamaResponse.tool_calls.length,
      });
      toolLogger.debug('Detailed tool_calls content.', { ...loggerContext, tool_calls: ollamaResponse.tool_calls });

      const mappedFunctionCalls: Array<import('@google/genai').FunctionCall> = [];

      parts = ollamaResponse.tool_calls.map((toolCall) => {
        const functionCallValue: import('@google/genai').FunctionCall = {
          name: toolCall.function.name,
          args: toolCall.function.parameters, // Ollama uses 'parameters', Gemini uses 'args'
        };
        mappedFunctionCalls.push(functionCallValue);
        toolLogger.trace('Mapped Ollama tool_call to Gemini FunctionCall part.', { ...loggerContext, mapped_function_call: functionCallValue });
        return { functionCall: functionCallValue };
      });

      topLevelFunctionCalls = mappedFunctionCalls;
      text = undefined; // Per Gemini spec, if functionCall is present, text content is typically not.
    } else {
      parts = ollamaResponse.response
        ? [{ text: ollamaResponse.response }]
        : [];
      text = ollamaResponse.response;
      if (text) {
        toolLogger.trace('Received text response from Ollama.', { ...loggerContext, response_text_length: text.length });
      }
    }

    const candidates: Array<import('@google/genai').Candidate> = [
      {
        content: {
          parts,
          role: 'model',
        },
        finishReason,
        index: 0,
        safetyRatings,
        tokenCount: ollamaResponse.eval_count,
      },
    ];

    const geminiResponse: GenerateContentResponse = {
      candidates,
      promptFeedback: {
        safetyRatings,
      },
      text,
      functionCalls: topLevelFunctionCalls,
      data: undefined,
      executableCode: undefined,
      codeExecutionResult: undefined,
    };

    return geminiResponse;
  }

  async generateContent(
    request: ExtendedGenerateContentParameters, // Use extended interface
  ): Promise<GenerateContentResponse> {
    const loggerContext = {
      generator: 'OllamaContentGenerator',
      method: 'generateContent',
      model: this.modelName,
    };
    toolLogger.info('Received generateContent request.', loggerContext);
    toolLogger.debug('Full generateContent request details.', { ...loggerContext, request });

    const ollamaParams = this.paramsFromGeminiRequest(request);
    ollamaParams.stream = false;
    toolLogger.debug('Parameters prepared for non-streaming Ollama call.', { ...loggerContext, ollamaParams });

    try {
      // The ollamaClient.generate method now includes a requestId in its logs.
      // We can capture it if we modify generate to return it, or just rely on correlated logging.
      // For now, no direct requestId is passed to responseToGeminiResponse.
      const ollamaResponse = (await this.ollamaClient.generate(
        ollamaParams,
      )) as OllamaGenerateResponse;

      toolLogger.info('Received non-streaming response from Ollama client.', {
        ...loggerContext,
        ollama_response_done: ollamaResponse.done,
        has_tool_calls: !!(ollamaResponse.tool_calls && ollamaResponse.tool_calls.length > 0),
      });
      return this.responseToGeminiResponse(ollamaResponse);
    } catch (error) {
      toolLogger.error('Error during non-streaming Ollama generate call.', { ...loggerContext, error, ollamaParams });
      throw error;
    }
  }

  async generateContentStream(
    request: ExtendedGenerateContentParameters, // Use extended interface
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const loggerContext = {
      generator: 'OllamaContentGenerator',
      method: 'generateContentStream',
      model: this.modelName,
    };
    toolLogger.info('Received generateContentStream request.', loggerContext);
    toolLogger.debug('Full generateContentStream request details.', { ...loggerContext, request });

    const ollamaParams = this.paramsFromGeminiRequest(request);
    ollamaParams.stream = true;
    toolLogger.debug('Parameters prepared for streaming Ollama call.', { ...loggerContext, ollamaParams });

    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const self = this;
    let streamOverallRequestId: string | undefined; // To potentially capture requestId from ollamaClient if it were returned by generate()

    async function* geminiStream(): AsyncGenerator<GenerateContentResponse> {
      let finalOllamaResponse: OllamaGenerateResponse | null = null;
      let chunkCount = 0;

      try {
        const stream = (await self.ollamaClient.generate(
          ollamaParams,
        )) as AsyncGenerator<OllamaGenerateResponse>;

        for await (const ollamaChunk of stream) {
          chunkCount++;
          // If ollamaClient.generate could return requestId, we'd use it here.
          // For now, logs from ollamaClient will have their own, logs here will use loggerContext.
          const chunkLoggerContext = { ...loggerContext, chunk: chunkCount, ollama_chunk_done: ollamaChunk.done, has_tool_calls: !!(ollamaChunk.tool_calls && ollamaChunk.tool_calls.length > 0) };
          toolLogger.debug('Processing Ollama stream chunk.', chunkLoggerContext);

          if (ollamaChunk.tool_calls && ollamaChunk.tool_calls.length > 0) {
            toolLogger.info('Tool calls received in stream chunk.', { ...chunkLoggerContext, tool_calls: ollamaChunk.tool_calls });
            // Handling of tool calls in stream is complex. Gemini typically expects them as a single block.
            // For now, we will map them if they appear in a chunk.
          }

          finalOllamaResponse = ollamaChunk;

          const currentText = ollamaChunk.response;
          let partsForChunk: Array<import('@google/genai').Part> = currentText ? [{ text: currentText }] : [];
          let functionCallsInChunk: Array<import('@google/genai').FunctionCall> | undefined = undefined;

          if (ollamaChunk.tool_calls && ollamaChunk.tool_calls.length > 0) {
            const mappedFCs: Array<import('@google/genai').FunctionCall> = [];
            partsForChunk = ollamaChunk.tool_calls.map(tc => {
              const fc: import('@google/genai').FunctionCall = { name: tc.function.name, args: tc.function.parameters };
              mappedFCs.push(fc);
              return { functionCall: fc };
            });
            functionCallsInChunk = mappedFCs;
          }

          const partialGeminiResponse: GenerateContentResponse = {
            candidates: [
              {
                content: { parts: partsForChunk, role: 'model' },
                finishReason: ollamaChunk.done ? self.mapDoneReason(ollamaChunk.done_reason) : FinishReason.FINISH_REASON_UNSPECIFIED,
                index: 0,
                safetyRatings: [{ category: HarmCategory.HARM_CATEGORY_UNSPECIFIED, probability: HarmProbability.NEGLIGIBLE }],
                tokenCount: ollamaChunk.eval_count,
              },
            ],
            promptFeedback: {
              safetyRatings: [{ category: HarmCategory.HARM_CATEGORY_UNSPECIFIED, probability: HarmProbability.NEGLIGIBLE }],
            },
            text: functionCallsInChunk ? undefined : currentText,
            functionCalls: functionCallsInChunk,
            data: undefined, executableCode: undefined, codeExecutionResult: undefined,
          };

          toolLogger.trace('Yielding partial Gemini response from stream.', { ...chunkLoggerContext, partialGeminiResponse });
          yield partialGeminiResponse;

          if (ollamaChunk.done) {
            toolLogger.info('Ollama stream marked as done.', { ...chunkLoggerContext, reason: ollamaChunk.done_reason });
            break;
          }
        }
      } catch (error) {
        toolLogger.error('Error during Ollama generate stream call.', { ...loggerContext, error, ollamaParams });
        throw error;
      }

      if (finalOllamaResponse && finalOllamaResponse.context) {
        self.currentContext = finalOllamaResponse.context;
        toolLogger.debug('Updated Ollama context from final stream response.', { ...loggerContext, context_length: finalOllamaResponse.context.length });
      }
      toolLogger.info('Gemini stream processing complete.', { ...loggerContext, total_chunks: chunkCount });
    }
    return geminiStream();
  }

  private mapDoneReason(doneReason?: string): FinishReason {
    if (!doneReason) return FinishReason.FINISH_REASON_UNSPECIFIED;
    switch (doneReason) {
      case 'stop':
        return FinishReason.STOP;
      case 'length':
        return FinishReason.MAX_TOKENS;
      default:
        return FinishReason.OTHER;
    }
  }

  async countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    // Ollama API does not have a dedicated /api/tokenize or /api/count_tokens endpoint.
    // Token counting would typically be done client-side using the model's specific tokenizer.
    // This is complex to implement generically here as it requires knowing the exact tokenizer for `this.modelName`.
    //
    // For now, we will return a placeholder and log a warning.
    // A more advanced implementation might:
    // 1. Attempt to use a library like `gpt-tokenizer` or `tiktoken` if the model is known to be compatible.
    // 2. Allow users to configure a tokenizer or provide token counts.
    // 3. The `generate` endpoint in Ollama *does* return `prompt_eval_count` and `eval_count`
    //    which are token counts, but that requires actually making a generation call.

    let numChars = 0;
    if (typeof request.contents === 'string') {
      numChars = request.contents.length;
    } else if (Array.isArray(request.contents)) {
      // request.contents is Array<string | Content>
      request.contents.forEach((contentItem) => {
        // contentItem is string | Content
        // Ensure contentItem is a Content object
        if (
          typeof contentItem !== 'string' &&
          'parts' in contentItem &&
          Array.isArray(contentItem.parts)
        ) {
          contentItem.parts.forEach((part: import('@google/genai').Part) => {
            if (part.text && typeof part.text === 'string') {
              numChars += part.text.length;
            }
          });
        } else if (typeof contentItem === 'string') {
          // This case should ideally not happen if CountTokensParameters expects Content objects for array items
          // but handling it defensively if request.contents can be Array<string>
          numChars += contentItem.length;
        }
      });
    } else if (
      request.contents &&
      typeof request.contents !== 'string' &&
      'parts' in request.contents &&
      Array.isArray(request.contents.parts)
    ) {
      // Handle single Content object (request.contents is Content)
      request.contents.parts.forEach((part: import('@google/genai').Part) => {
        if (part.text && typeof part.text === 'string') {
          numChars += part.text.length;
        }
      });
    }

    // Extremely rough heuristic: average 3-4 characters per token. Let's use 3.5.
    // This is NOT accurate and should be replaced if a better method is found.
    const estimatedTokens = Math.ceil(numChars / 3.5);

    toolLogger.warn(`Ollama countTokens is using a very rough character-based estimation for model "${this.modelName}". This is not accurate.`, {
      generator: 'OllamaContentGenerator',
      method: 'countTokens',
      model: this.modelName,
      request_contents_type: typeof request.contents,
      estimatedChars: numChars,
      estimatedTokens,
    });

    return { totalTokens: estimatedTokens };
  }

  async embedContent(
    request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    const loggerContext = {
      generator: 'OllamaContentGenerator',
      method: 'embedContent',
      model: this.modelName,
    };
    toolLogger.info('Received embedContent request.', loggerContext);
    toolLogger.debug('Full embedContent request details.', { ...loggerContext, request });

    // Extract prompt from EmbedContentParameters
    // Similar to generateContent, this is a simplified extraction
    let promptText = '';
    if (typeof request.contents === 'string') {
      promptText = request.contents;
    } else if (Array.isArray(request.contents)) {
      // request.contents is Array<string | Content>
      // Handle array of Content objects, taking the last one
      const lastContentItem = request.contents[request.contents.length - 1]; // lastContentItem is string | Content
      // Ensure lastContentItem is a Content object
      if (
        typeof lastContentItem !== 'string' &&
        'parts' in lastContentItem &&
        Array.isArray(lastContentItem.parts)
      ) {
        promptText = lastContentItem.parts
          .map((part: import('@google/genai').Part) => part.text || '')
          .join(' ');
      }
      // If lastContentItem is a string, it's not directly handled here for promptText from an array.
      // The original logic seemed to imply only Content objects with parts contribute to promptText from an array.
    } else if (
      request.contents &&
      typeof request.contents !== 'string' &&
      'parts' in request.contents &&
      Array.isArray(request.contents.parts)
    ) {
      // Handle single Content object (request.contents is Content)
      promptText = request.contents.parts
        .map((part: import('@google/genai').Part) => part.text || '')
        .join(' ');
    }

    if (!promptText) {
      toolLogger.error('Prompt text is required for Ollama embedContent but was not found/extracted.', { ...loggerContext, request });
      throw new Error('Prompt text is required for Ollama embedContent.');
    }

    const ollamaParams: OllamaEmbeddingsParams = {
      model: this.modelName,
      prompt: promptText,
    };
    toolLogger.debug('Parameters prepared for Ollama embeddings call.', { ...loggerContext, ollamaParams });

    try {
      const ollamaResponse: OllamaEmbeddingsResponse =
        await this.ollamaClient.embeddings(ollamaParams);
      toolLogger.info('Received embeddings from Ollama client.', { ...loggerContext, embedding_length: ollamaResponse.embedding?.length });

      return {
        embeddings: [ { values: ollamaResponse.embedding } ],
      };
    } catch (error) {
      toolLogger.error('Error during Ollama embeddings call.', { ...loggerContext, error, ollamaParams });
      throw error;
    }
  }
}

import { Config } from '../config/config.js'; // Import Config

export async function createContentGenerator(
  contentGeneratorConfig: ContentGeneratorConfig,
  // Add full config object to allow OllamaClient to be configured (e.g. with baseUrl from settings)
  // This was missing and is crucial for proper OllamaClient instantiation.
  config: Config,
): Promise<ContentGenerator> {
  const version = process.env.CLI_VERSION || process.version;
  const httpOptions = {
    headers: {
      'User-Agent': `GeminiCLI/${version} (${process.platform}; ${process.arch})`,
    },
  };

  if (contentGeneratorConfig.authType === AuthType.USE_OLLAMA) {
    // TODO: The baseUrl for OllamaClient should ideally come from config settings,
    // not be hardcoded or only rely on the default in OllamaClient constructor.
    // For now, we assume the default 'http://localhost:11434' or that Config might provide it.
    // A more robust solution would be to have `config.getOllamaBaseUrl()`
    const ollamaClient = new OllamaClient(config); // Pass the main Config object
    console.log(
      `Creating OllamaContentGenerator for model: ${contentGeneratorConfig.model}`,
    );
    return new OllamaContentGenerator(
      contentGeneratorConfig.model,
      ollamaClient,
    );
  }

  if (contentGeneratorConfig.authType === AuthType.LOGIN_WITH_GOOGLE_PERSONAL) {
    return createCodeAssistContentGenerator(
      httpOptions,
      contentGeneratorConfig.authType,
    );
  }

  if (
    contentGeneratorConfig.authType === AuthType.USE_GEMINI ||
    contentGeneratorConfig.authType === AuthType.USE_VERTEX_AI
  ) {
    const googleGenAI = new GoogleGenAI({
      apiKey:
        contentGeneratorConfig.apiKey === ''
          ? undefined
          : contentGeneratorConfig.apiKey,
      vertexai: contentGeneratorConfig.vertexai,
      httpOptions,
    });

    return googleGenAI.models;
  }

  throw new Error(
    `Error creating contentGenerator: Unsupported authType: ${contentGeneratorConfig.authType}`,
  );
}
