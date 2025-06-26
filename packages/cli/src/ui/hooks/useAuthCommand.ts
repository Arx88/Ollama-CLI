/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useCallback, useEffect } from 'react';
import { LoadedSettings, SettingScope } from '../../config/settings.js';
import {
  AuthType, // Remove one of the duplicate AuthType imports
  Config,
  clearCachedCredentialFile,
  getErrorMessage,
  OllamaClient, // Assuming OllamaClient is exported from core
} from '@google/gemini-cli-core';

async function performAuthFlow(
  authMethod: AuthType,
  config: Config,
  settings: LoadedSettings, // Added settings to access ollamaModel
  openAuthDialog: () => void, // To re-open dialog for model selection if needed
  setAuthError: (error: string | null) => void, // To display errors
) {
  if (authMethod === AuthType.USE_OLLAMA) {
    try {
      console.log('Performing Ollama authentication...');
      const ollamaClient = new OllamaClient(config); // Assumes config can be passed
      const availableModels = await ollamaClient.listModels();

      if (availableModels.models.length === 0) {
        setAuthError(
          'No Ollama models found. Please ensure Ollama is running and models are installed.',
        );
        openAuthDialog(); // Re-open to allow choosing another method or retrying
        return;
      }

      let preferredModelName =
        settings.merged.ollamaModel || config.getOllamaModel();
      let finalModelToUse: string | undefined = undefined;

      if (availableModels.models.length === 0) {
        setAuthError(
          'No Ollama models found. Please ensure Ollama is running and models are installed.',
        );
        openAuthDialog(); // Re-open to allow choosing another method or retrying
        return;
      }

      if (preferredModelName) {
        // 1. Try exact match first
        const exactMatch = availableModels.models.find(
          (m) => m.name === preferredModelName,
        );
        if (exactMatch) {
          finalModelToUse = exactMatch.name;
        } else {
          // 2. Try flexible match (prefix + ':' or direct match if preferredModelName already has a tag)
          const flexibleMatches = availableModels.models.filter(
            (m) =>
              m.name === preferredModelName || // Case: preferredModelName already has tag
              m.name.startsWith(preferredModelName + ':'), // Case: preferredModelName is base
          );

          if (flexibleMatches.length > 0) {
            // Prefer a ':latest' tag if available among flexible matches
            const latestTagMatch = flexibleMatches.find((m) =>
              m.name.endsWith(':latest'),
            );
            if (latestTagMatch) {
              finalModelToUse = latestTagMatch.name;
            } else {
              // Otherwise, use the first flexible match (e.g., model:tag123)
              finalModelToUse = flexibleMatches[0].name;
            }
            console.log(
              `Using Ollama model "${finalModelToUse}" based on preferred model "${preferredModelName}".`,
            );
          }
        }
      }

      // 3. If no model could be determined from settings/config, or if the preferred model wasn't found
      if (!finalModelToUse) {
        if (preferredModelName) {
          // Only log warning if a preferred model was set but not found after flexible search
          console.warn(
            `Preferred Ollama model "${preferredModelName}" not found among available models. Falling back.`,
          );
        }
        // Fallback to the first model in the list (using its full name including tag)
        finalModelToUse = availableModels.models[0].name;
        console.log(
          `Using first available Ollama model by default: ${finalModelToUse}`,
        );
      }

      // Set the chosen model in the current session's config
      config.setOllamaModel(finalModelToUse);
      // Persist this choice in settings only if it came from fallback and no prior setting existed,
      // OR if the flexible match resolved to a more specific name (e.g. with tag)
      // This avoids overwriting a user's base model preference (e.g., "llama3") with "llama3:latest"
      // unless it was the only way to resolve it.
      // However, the current settings mechanism in useOllamaModelSelection already handles saving the user's explicit choice.
      // So, we primarily ensure config is updated here for the session.
      // If `performAuthFlow` is called *without* user interaction (e.g. initial load),
      // and `settings.merged.ollamaModel` was empty, then `useOllamaModelSelection`'s dialog
      // should subsequently open if `finalModelToUse` is set, allowing user to confirm/change.
      // The main goal here is that `config.setOllamaModel` has the *actually usable* model name.
      // This part of refreshAuth will need to be adapted for Ollama
      // to correctly set up the content generator for an Ollama model.
      await config.refreshAuth(authMethod);
      console.log(`Authenticated via Ollama with model ${selectedModel}.`);
      setAuthError(null); // Clear any previous auth errors
    } catch (e) {
      const errorMessage = getErrorMessage(e);
      console.error('Ollama authentication failed:', errorMessage);
      setAuthError(
        `Ollama authentication failed: ${errorMessage}. Please ensure Ollama is running and accessible.`,
      );
      openAuthDialog(); // Re-open to allow choosing another method or retrying
    }
  } else {
    await config.refreshAuth(authMethod);
    console.log(`Authenticated via "${authMethod}".`);
  }
}

export const useAuthCommand = (
  settings: LoadedSettings,
  setAuthError: (error: string | null) => void,
  config: Config,
) => {
  const [isAuthDialogOpen, setIsAuthDialogOpen] = useState(
    settings.merged.selectedAuthType === undefined,
  );

  const openAuthDialog = useCallback(() => {
    setIsAuthDialogOpen(true);
  }, []);

  const [isAuthenticating, setIsAuthenticating] = useState(false);

  useEffect(() => {
    const authFlow = async () => {
      if (isAuthDialogOpen || !settings.merged.selectedAuthType) {
        return;
      }

      try {
        setIsAuthenticating(true);
        await performAuthFlow(
          settings.merged.selectedAuthType as AuthType,
          config,
          settings,
          openAuthDialog,
          setAuthError,
        );
      } catch (e) {
        // This catch block might be redundant if performAuthFlow handles its own errors
        // and calls setAuthError/openAuthDialog. However, keeping it for general errors.
        let errorMessage = `Failed to login.\nMessage: ${getErrorMessage(e)}`;
        if (
          settings.merged.selectedAuthType ===
            AuthType.LOGIN_WITH_GOOGLE_PERSONAL &&
          !process.env.GOOGLE_CLOUD_PROJECT
        ) {
          errorMessage =
            'Failed to login. Workspace accounts and licensed Code Assist users must configure' +
            ` GOOGLE_CLOUD_PROJECT (see https://goo.gle/gemini-cli-auth-docs#workspace-gca).\nMessage: ${getErrorMessage(e)}`;
        }
        setAuthError(errorMessage);
        openAuthDialog();
      } finally {
        setIsAuthenticating(false);
      }
    };

    void authFlow();
  }, [isAuthDialogOpen, settings, config, setAuthError, openAuthDialog]);

  const handleAuthSelect = useCallback(
    async (authMethod: string | undefined, scope: SettingScope) => {
      if (authMethod) {
        await clearCachedCredentialFile();
        settings.setValue(scope, 'selectedAuthType', authMethod);
      }
      setIsAuthDialogOpen(false);
      setAuthError(null);
    },
    [settings, setAuthError],
  );

  const handleAuthHighlight = useCallback((_authMethod: string | undefined) => {
    // For now, we don't do anything on highlight.
  }, []);

  const cancelAuthentication = useCallback(() => {
    setIsAuthenticating(false);
  }, []);

  return {
    isAuthDialogOpen,
    openAuthDialog,
    handleAuthSelect,
    handleAuthHighlight,
    isAuthenticating,
    cancelAuthentication,
  };
};
