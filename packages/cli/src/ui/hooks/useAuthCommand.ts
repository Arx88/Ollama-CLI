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

      // For now, let's try to use the model from settings or default,
      // or the first available if not set.
      // A proper model selection UI would be better.
      let selectedModel = settings.merged.ollamaModel || config.getOllamaModel();

      if (
        !selectedModel ||
        !availableModels.models.find((m) => m.name === selectedModel)
      ) {
        if (selectedModel) {
          console.warn(
            `Previously selected Ollama model "${selectedModel}" not found. Falling back.`,
          );
        }
        selectedModel = availableModels.models[0].name.split(':')[0]; // Use the base name of the first model
        console.log(`Using first available Ollama model: ${selectedModel}`);
      }

      // We need a way to store this selected model persistently if it changed.
      // For now, just set it in the current session's config.
      config.setOllamaModel(selectedModel);
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
