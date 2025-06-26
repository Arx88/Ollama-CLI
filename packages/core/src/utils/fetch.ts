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

// Original function name and signature
export async function fetchWithTimeout(
  url: string,
  timeout: number, // Original signature had timeout as a direct parameter
  options?: RequestInit, // Options can be passed but signal for timeout is handled internally
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    // Combine passed options with the internal signal for timeout
    const fetchOptions = { ...options, signal: controller.signal };
    const response = await fetch(url, fetchOptions);
    return response;
  } catch (error) {
    if (isNodeError(error) && error.code === 'ABORT_ERR') {
      // This specific error is thrown by fetch when our timeout aborts the request
      throw new FetchError(`Request timed out after ${timeout}ms`, 'ETIMEDOUT');
    }
    // For other errors, wrap them in FetchError
    throw new FetchError(getErrorMessage(error));
  } finally {
    clearTimeout(timeoutId);
  }
}
