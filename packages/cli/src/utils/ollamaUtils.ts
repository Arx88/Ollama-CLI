/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

// Assuming global fetch is available (Node 18+)
// import fetch from 'node-fetch'; // Removed: node-fetch is not a direct dependency

// Interfaces for Ollama API response
interface OllamaModelDetails {
  format: string;
  family: string;
  families: string[] | null;
  parameter_size: string;
  quantization_level: string;
}

interface OllamaModelTag {
  name: string;
  modified_at: string;
  size: number;
  digest: string;
  details: OllamaModelDetails;
}

interface OllamaApiTagsResponse {
  models: OllamaModelTag[];
}

/**
 * Fetches the list of available models from an Ollama API endpoint.
 * @param ollamaApiEndpoint The base URL of the Ollama API (e.g., "http://localhost:11434")
 * @returns A promise that resolves to an array of model name strings, or an empty array if an error occurs.
 */
export async function getOllamaModels(
  ollamaApiEndpoint: string,
): Promise<string[]> {
  try {
    const response = await fetch(`${ollamaApiEndpoint}/api/tags`);
    if (!response.ok) {
      // LogToFile could be used here if integrated, or throw a specific error
      console.error(
        `Ollama API request failed: ${response.status} ${response.statusText}`,
      );
      // Try to get error message from Ollama if possible
      try {
        const errorData = await response.json();
        console.error('Ollama API error details:', errorData);
      } catch (_e) {
        // Ignore if error response is not JSON
      }
      return [];
    }

    const data = (await response.json()) as OllamaApiTagsResponse;

    if (data && Array.isArray(data.models)) {
      return data.models.map((model) => model.name);
    } else {
      console.error('Unexpected response format from Ollama /api/tags');
      return [];
    }
  } catch (error) {
    console.error(`Error fetching Ollama models: ${error}`);
    return [];
  }
}
