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
  HarmBlockThreshold,
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
  config?: { getModel?: () => string },
): Promise<ContentGeneratorConfig> {
  const geminiApiKey = process.env.GEMINI_API_KEY;
  const googleApiKey = process.env.GOOGLE_API_KEY;
  const googleCloudProject = process.env.GOOGLE_CLOUD_PROJECT;
  const googleCloudLocation = process.env.GOOGLE_CLOUD_LOCATION;
  // Ollama specific config
  const ollamaModel = (config as any)?.getOllamaModel?.() || process.env.OLLAMA_MODEL; // Type assertion

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

// Placeholder for Ollama Content Generator
// This would need to implement the ContentGenerator interface
// and interact with the Ollama API (likely via OllamaClient).
export class OllamaContentGenerator implements ContentGenerator { // Added export here
  private readonly modelName: string;
  private readonly ollamaClient: OllamaClient;
  private currentContext: number[] | undefined; // To store context for conversational turns

  constructor(modelName: string, ollamaClient: OllamaClient) {
    this.modelName = modelName;
    this.ollamaClient = ollamaClient;
    if (process.env.GEMINI_DEBUG === 'true') {
      console.log(
        `OllamaContentGenerator initialized with model: ${this.modelName}`,
      );
    }
  }

  private paramsFromGeminiRequest(
    request: GenerateContentParameters,
  ): OllamaGenerateParams {
    // TODO: More sophisticated mapping, especially for 'contents' (multi-turn chat history)
    // For now, concatenate text parts from the last user message.
    let promptText = '';
    if (Array.isArray(request.contents) && request.contents.length > 0) {
      const lastContentItem = request.contents[request.contents.length - 1]; // Type: string | Content
      // Ensure lastContentItem is a Content object (has 'role' and 'parts')
      if (typeof lastContentItem !== 'string' && 'parts' in lastContentItem && Array.isArray(lastContentItem.parts)) {
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
     // TODO: Handle system instruction if provided in request.contents
     // (e.g. first message with role 'system' or a dedicated field if available)

    return {
      model: this.modelName,
      prompt: promptText,
      system: undefined, // Placeholder for system prompt
      template: undefined, // Placeholder for template
      context: this.currentContext, // Pass context from previous turn
      stream: false, // Explicitly false for this method
      options: Object.keys(options).length > 0 ? options : undefined,
    };
  }

  private responseToGeminiResponse(
    ollamaResponse: OllamaGenerateResponse,
  ): GenerateContentResponse {
    // Store context for next turn
    if (ollamaResponse.context) {
      this.currentContext = ollamaResponse.context;
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
      // HarmBlockThreshold.BLOCK_NONE is not assignable to HarmProbability.
      // Using HarmProbability.NEGLIGIBLE as a stand-in.
      // This might need adjustment based on how BLOCK_NONE should be interpreted.
      { category: HarmCategory.HARM_CATEGORY_UNSPECIFIED, probability: HarmProbability.NEGLIGIBLE },
    ];


    return {
      // text, functionCalls etc. are properties of GenerateContentResponse, not methods
      text: ollamaResponse.response,
      functionCalls: undefined,
      executableCode: undefined,
      codeExecutionResult: undefined,
      data: "", // Changed to empty string
      candidates: [
        {
          content: {
            parts: [{ text: ollamaResponse.response }],
            role: 'model',
          },
          finishReason: finishReason,
          index: 0,
          // safetyRatings: [], // Ollama doesn't provide these directly
          // citationMetadata: undefined, // Ollama doesn't provide this
          tokenCount: ollamaResponse.eval_count, // Approximate token count
          safetyRatings: safetyRatings,
        },
      ],
      promptFeedback: {
        // blockReason: undefined, // Ollama doesn't provide this
        safetyRatings: safetyRatings, // Placeholder
      },
      // usageMetadata: { // Not directly available, but can be constructed if needed
      //   promptTokenCount: ollamaResponse.prompt_eval_count,
      //   candidatesTokenCount: ollamaResponse.eval_count,
      //   totalTokenCount: (ollamaResponse.prompt_eval_count || 0) + (ollamaResponse.eval_count || 0)
      // }
    };
  }

  async generateContent(
    request: GenerateContentParameters,
  ): Promise<GenerateContentResponse> {
    const ollamaParams = this.paramsFromGeminiRequest(request);
    ollamaParams.stream = false; // Ensure it's non-streaming

    const ollamaResponse = (await this.ollamaClient.generate(
      ollamaParams,
    )) as OllamaGenerateResponse; // Type assertion for non-streaming

    return this.responseToGeminiResponse(ollamaResponse);
  }

  async generateContentStream(
    request: GenerateContentParameters,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const ollamaParams = this.paramsFromGeminiRequest(request);
    ollamaParams.stream = true; // Ensure it's streaming

    const stream = (await this.ollamaClient.generate(
      ollamaParams,
    )) as AsyncGenerator<OllamaGenerateResponse>; // Type assertion for streaming

    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const self = this; // To access 'this' inside the async generator
    async function* geminiStream(): AsyncGenerator<GenerateContentResponse> {
      let accumulatedResponse = '';
      let finalOllamaResponse: OllamaGenerateResponse | null = null;

      for await (const ollamaChunk of stream) {
        accumulatedResponse += ollamaChunk.response;
        finalOllamaResponse = ollamaChunk; // Keep track of the latest chunk for final context

        // Yield a Gemini-formatted response for each chunk
        // Note: Ollama's streaming gives partial 'response' in each chunk.
        // The 'done' field indicates the end of the full response.
        // We need to decide how to map this to Gemini's streaming, which might expect
        // a full candidate part in each yielded response or cumulative.
        // For simplicity, we'll yield cumulative text for now, but only the new part.

        const partialGeminiResponse: GenerateContentResponse = {
          candidates: [
            {
              content: {
                parts: [{ text: ollamaChunk.response }], // Text of the current chunk
                role: 'model',
              },
              finishReason: ollamaChunk.done
                ? self.mapDoneReason(ollamaChunk.done_reason)
                : FinishReason.FINISH_REASON_UNSPECIFIED,
              index: 0,
              safetyRatings: [ // Default safety ratings
                // HarmBlockThreshold.BLOCK_NONE is not assignable to HarmProbability.
                // Using HarmProbability.NEGLIGIBLE as a stand-in.
                { category: HarmCategory.HARM_CATEGORY_UNSPECIFIED, probability: HarmProbability.NEGLIGIBLE },
              ],
              tokenCount: ollamaChunk.eval_count, // Tokens for this chunk if available
            },
          ],
          promptFeedback: { // Placeholder
            safetyRatings: [
                // HarmBlockThreshold.BLOCK_NONE is not assignable to HarmProbability.
                // Using HarmProbability.NEGLIGIBLE as a stand-in.
              { category: HarmCategory.HARM_CATEGORY_UNSPECIFIED, probability: HarmProbability.NEGLIGIBLE },
            ],
          },
          // text, functionCalls etc. are properties of GenerateContentResponse, not methods
          text: ollamaChunk.response,
          functionCalls: undefined,
          executableCode: undefined,
          codeExecutionResult: undefined,
          data: "", // Changed to empty string
        };
        yield partialGeminiResponse;

        if (ollamaChunk.done) {
          break;
        }
      }

      // After the stream is done, update the context from the final response
      if (finalOllamaResponse && finalOllamaResponse.context) {
        self.currentContext = finalOllamaResponse.context;
      }
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
    } else if (Array.isArray(request.contents)) { // request.contents is Array<string | Content>
      request.contents.forEach(contentItem => { // contentItem is string | Content
        // Ensure contentItem is a Content object
        if (typeof contentItem !== 'string' && 'parts' in contentItem && Array.isArray(contentItem.parts)) {
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
    } else if (request.contents && typeof request.contents !== 'string' && 'parts' in request.contents && Array.isArray(request.contents.parts)) {
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

    console.warn(
      `Ollama countTokens is using a very rough character-based estimation for model "${this.modelName}". ` +
      `Request: ${JSON.stringify(request)}. Estimated chars: ${numChars}, Estimated tokens: ${estimatedTokens}. ` +
      `This is not accurate.`,
    );

    return { totalTokens: estimatedTokens };
  }

  async embedContent(
    request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    // Extract prompt from EmbedContentParameters
    // Similar to generateContent, this is a simplified extraction
    let promptText = '';
    if (typeof request.contents === 'string') {
      promptText = request.contents;
    } else if (Array.isArray(request.contents)) { // request.contents is Array<string | Content>
      // Handle array of Content objects, taking the last one
      const lastContentItem = request.contents[request.contents.length - 1]; // lastContentItem is string | Content
      // Ensure lastContentItem is a Content object
      if (typeof lastContentItem !== 'string' && 'parts' in lastContentItem && Array.isArray(lastContentItem.parts)) {
        promptText = lastContentItem.parts
          .map((part: import('@google/genai').Part) => part.text || '')
          .join(' ');
      }
      // If lastContentItem is a string, it's not directly handled here for promptText from an array.
      // The original logic seemed to imply only Content objects with parts contribute to promptText from an array.
    } else if (request.contents && typeof request.contents !== 'string' && 'parts' in request.contents && Array.isArray(request.contents.parts)) {
      // Handle single Content object (request.contents is Content)
      promptText = request.contents.parts
        .map((part: import('@google/genai').Part) => part.text || '')
        .join(' ');
    }

    if (!promptText) {
      throw new Error('Prompt text is required for Ollama embedContent.');
    }

    const ollamaParams: OllamaEmbeddingsParams = {
      model: this.modelName,
      prompt: promptText,
      // options: undefined, // Add if specific options are needed for embeddings
    };

    const ollamaResponse: OllamaEmbeddingsResponse =
      await this.ollamaClient.embeddings(ollamaParams);

    return {
      embedding: {
        values: ollamaResponse.embedding,
      },
    } as EmbedContentResponse; // Cast to EmbedContentResponse, though structure matches
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
    return createCodeAssistContentGenerator(httpOptions, contentGeneratorConfig.authType);
  }

  if (
    contentGeneratorConfig.authType === AuthType.USE_GEMINI ||
    contentGeneratorConfig.authType === AuthType.USE_VERTEX_AI
  ) {
    const googleGenAI = new GoogleGenAI({
      apiKey: contentGeneratorConfig.apiKey === '' ? undefined : contentGeneratorConfig.apiKey,
      vertexai: contentGeneratorConfig.vertexai,
      httpOptions,
    });

    return googleGenAI.models;
  }

  throw new Error(
    `Error creating contentGenerator: Unsupported authType: ${contentGeneratorConfig.authType}`,
  );
}
