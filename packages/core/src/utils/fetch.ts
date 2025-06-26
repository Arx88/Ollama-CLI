/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { getErrorMessage, isNodeError } from './errors.js';
import { URL } from 'url';

const PRIVATE_IP_RANGES = [
  /^10\./,
  /^127\./,
  /^172\.(1[6-9]|2[0-9]|3[0-1])\./,
  /^192\.168\./,
  /^::1$/,
  /^fc00:/,
  /^fe80:/,
];

export class FetchError extends Error {
  constructor(
    message: string,
    public code?: string,
  ) {
    super(message);
    this.name = 'FetchError';
  }
}

export function isPrivateIp(url: string): boolean {
  try {
    const hostname = new URL(url).hostname;
    return PRIVATE_IP_RANGES.some((range) => range.test(hostname));
  } catch (_e) {
    return false;
  }
}

// Renaming to fetchWithRetry to match OllamaClient's expectation.
// Actual retry logic is not implemented here yet.
export async function fetchWithRetry(
  url: string,
  options?: RequestInit, // Added options to be compatible with typical fetch usage
  timeout = 10000, // Default timeout, can be overridden by options.signal
): Promise<Response> {
  const controller = new AbortController();
  let timeoutId: NodeJS.Timeout | undefined;

  // If a signal is provided in options, prefer it. Otherwise, use our timeout.
  const signal = options?.signal ?? controller.signal;
  if (!options?.signal) {
    timeoutId = setTimeout(() => controller.abort(), timeout);
  }

  try {
    // Pass through all options to fetch
    const response = await fetch(url, { ...options, signal });
    return response;
  } catch (error) {
    if (isNodeError(error) && error.code === 'ABORT_ERR') {
      // Distinguish between external abort and our timeout
      if (timeoutId && controller.signal.aborted) {
         throw new FetchError(`Request timed out after ${timeout}ms`, 'ETIMEDOUT');
      }
      // If aborted by external signal, rethrow preserving original error if possible
      // or a generic abort error.
      throw new FetchError(getErrorMessage(error) || 'Request aborted', 'ABORT_ERR');
    }
    throw new FetchError(getErrorMessage(error));
  } finally {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
  }
}
