/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useCallback } from 'react';
import { getOllamaModels } from '../../utils/ollamaUtils.js';
import { type Config } from '@google/gemini-cli-core';
import { type LoadedSettings, SettingScope } from '../../config/settings.js';
import { logToFile } from '../../utils/fileLogger.js'; // For debugging this hook

export interface OllamaModelSelectionResult {
  isOllamaModelDialogOpen: boolean;
  isLoadingModels: boolean;
  availableModels: string[];
  errorLoadingModels: string | null;
  openOllamaModelDialog: () => Promise<void>;
  // handleModelSelect is internal to the hook now, triggered by dialog
  onModelSelectedFromDialog: (modelName: string) => void;
  handleDialogClose: () => void;
}

export function useOllamaModelSelection(
  config: Config,
  settings: LoadedSettings,
  // Callback to inform App.tsx that a model was selected
  onModelSelected?: (modelName: string) => void,
  onDialogCancelled?: () => void,
): OllamaModelSelectionResult {
  const [isOllamaModelDialogOpen, setIsOllamaModelDialogOpen] = useState(false);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [errorLoadingModels, setErrorLoadingModels] = useState<string | null>(
    null,
  );

  const openOllamaModelDialog = useCallback(async () => {
    setIsOllamaModelDialogOpen(true);
    setIsLoadingModels(true);
    setErrorLoadingModels(null);
    logToFile('[useOllamaModelSelection] Opening dialog, fetching models...');

    try {
      // TODO: The Ollama API endpoint should ideally come from config
      // For now, let's assume a default or that config.getOllamaApiEndpoint() exists and is suitable.
      // If config.getOllamaApiEndpoint() is not available in the core Config,
      // we might need to add it or pass it directly.
      // Let's assume `config.getOllamaApiEndpoint()` is a method we can add or use if it exists.
      // If not, we'll default to a common one for now.
      const ollamaEndpoint =
        config.getOllamaApiEndpoint() || 'http://localhost:11434';

      logToFile(
        `[useOllamaModelSelection] Using Ollama endpoint: ${ollamaEndpoint}`,
      );
      const models = await getOllamaModels(ollamaEndpoint);
      setAvailableModels(models);
      if (models.length === 0) {
        logToFile(
          '[useOllamaModelSelection] No models returned from getOllamaModels.',
        );
        // setErrorLoadingModels('No Ollama models found. Ensure Ollama is running and models are pulled.');
        // Dialog itself will show a message if models array is empty.
      } else {
        logToFile(
          `[useOllamaModelSelection] Models fetched: ${models.join(', ')}`,
        );
      }
    } catch (error) {
      logToFile(`[useOllamaModelSelection] Error fetching models: ${error}`);
      setErrorLoadingModels(
        error instanceof Error
          ? error.message
          : 'Failed to fetch Ollama models.',
      );
      setAvailableModels([]); // Ensure models list is empty on error
    } finally {
      setIsLoadingModels(false);
    }
  }, [config]);

  // This is the function that the OllamaModelDialog will call when a model is selected by the user.
  const internalHandleModelSelect = useCallback(
    (modelName: string) => {
      logToFile(
        `[useOllamaModelSelection] Model selected in dialog: ${modelName}`,
      );
      // settings.setValue is now handled by the callback in App.tsx
      setIsOllamaModelDialogOpen(false);
      if (onModelSelected) {
        onModelSelected(modelName); // Pass the selected model name to App.tsx
      }
    },
    [onModelSelected], // Removed settings from dependencies
  );

  const handleDialogClose = useCallback(() => {
    logToFile(
      '[useOllamaModelSelection] Dialog explicitly closed/cancelled by user.',
    );
    setIsOllamaModelDialogOpen(false);
    if (onDialogCancelled) {
      onDialogCancelled();
    }
  }, [onDialogCancelled]);

  return {
    isOllamaModelDialogOpen,
    isLoadingModels,
    availableModels,
    errorLoadingModels,
    openOllamaModelDialog,
    // Expose the internal handler for the dialog to call
    // This might be confusing. Let's rename it in the returned object for clarity
    // or pass it directly to the dialog when App.tsx renders it.
    // For now, App.tsx will get this function and pass it to the dialog.
    onModelSelectedFromDialog: internalHandleModelSelect, // Renamed for clarity in App.tsx if needed
    handleDialogClose, // This is for explicit cancellation (e.g. Esc from dialog)
  };
}
