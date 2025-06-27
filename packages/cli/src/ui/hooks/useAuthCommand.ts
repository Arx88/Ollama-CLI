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
  try {
    if (authMethod === AuthType.USE_OLLAMA) {
      console.log(
        '[useAuthCommand] Performing Ollama pre-authentication check...',
      );
      const ollamaClient = new OllamaClient(config);
      try {
        const availableModels = await ollamaClient.listModels();
        if (availableModels.models.length === 0) {
          setAuthError(
            'No Ollama models found. Please ensure Ollama is running and models are installed.',
          );
          openAuthDialog(); // Re-open to allow choosing another method or retrying
          return; // Stop further processing for Ollama here
        }
        // Success: Ollama is running and has models.
        console.log(
          '[useAuthCommand] Ollama is accessible and has models. App.tsx will trigger model selection.',
        );
        setAuthError(null); // Clear any previous auth errors.
      } catch (e) {
        const errorMessage = getErrorMessage(e);
        console.error(
          '[useAuthCommand] Ollama accessibility check failed:',
          errorMessage,
        );
        setAuthError(
          `Ollama check failed: ${errorMessage}. Please ensure Ollama is running and accessible.`,
        );
        openAuthDialog(); // Re-open to allow choosing another method or retrying
        return; // Stop further processing
      }
      // No config.refreshAuth here for Ollama; it happens after explicit model selection in App.tsx.
    } else {
      // For other auth methods, refreshAuth immediately.
      await config.refreshAuth(authMethod);
      console.log(`[useAuthCommand] Authenticated via "${authMethod}".`);
      setAuthError(null); // Clear any previous auth errors
    }
  } catch (e) {
    // This catch block handles errors from config.refreshAuth (for non-Ollama methods)
    // or any other unexpected errors during the flow.
    const errorMessage = getErrorMessage(e);
    console.error(
      `[useAuthCommand] Authentication flow failed for ${authMethod}:`,
      errorMessage,
    );
    setAuthError(`Authentication failed for ${authMethod}: ${errorMessage}.`);
    // For non-Ollama methods, or if Ollama check itself failed before calling openAuthDialog,
    // we might want to reopen. However, Ollama's specific catch already calls openAuthDialog.
    // This ensures openAuthDialog is called if refreshAuth for other methods fails.
    if (authMethod !== AuthType.USE_OLLAMA) {
      openAuthDialog();
    }
  }
}

export const useAuthCommand = (
  settings: LoadedSettings,
  setAuthError: (error: string | null) => void,
  config: Config,
  // Callback to signal that Ollama auth type was chosen and check was successful
  onOllamaAuthTypeSelectedAndChecked?: () => void,
  // State from App.tsx to prevent re-triggering model dialog if already open
  isOllamaModelDialogOpen?: boolean,
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

      // If Ollama is the selected type, we don't immediately authenticate further here.
      // The selection itself is the "first step". App.tsx will handle opening the model dialog.
      // performAuthFlow for Ollama will just do a pre-check.
      if (settings.merged.selectedAuthType === AuthType.USE_OLLAMA) {
        if (isAuthenticating) return; // Avoid re-triggering if already in pre-check
        try {
          setIsAuthenticating(true);
          await performAuthFlow(
            AuthType.USE_OLLAMA,
            config,
            settings, // settings is not strictly needed by performAuthFlow for Ollama anymore
            openAuthDialog,
            setAuthError,
          );
          // If performAuthFlow for Ollama was successful (it didn't throw or call openAuthDialog),
          // it means Ollama is accessible. Now signal to open model dialog,
          // but only if it's not already open (to break potential loops).
          if (
            settings.merged.selectedAuthType === AuthType.USE_OLLAMA &&
            !isAuthDialogOpen && // Ensure auth dialog is closed
            !isOllamaModelDialogOpen // Check if model dialog is already open
          ) {
            onOllamaAuthTypeSelectedAndChecked?.();
          }
        } catch (e) {
          // Should be caught by performAuthFlow's catch, but as a safeguard:
          const errMsg = getErrorMessage(e);
          setAuthError(`Ollama setup error: ${errMsg}`);
          openAuthDialog();
        } finally {
          setIsAuthenticating(false);
        }
        return; // Stop here for Ollama, App.tsx takes over for model selection
      }

      // For other auth types, proceed with the original flow
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
  }, [
    isAuthDialogOpen,
    settings,
    config,
    setAuthError,
    openAuthDialog,
    onOllamaAuthTypeSelectedAndChecked,
    isAuthenticating,
    isOllamaModelDialogOpen, // Added to dependencies
  ]);

  const handleAuthSelect = useCallback(
    async (authMethod: string | undefined, scope: SettingScope) => {
      if (authMethod) {
        const previousAuthType = settings.merged.selectedAuthType;
        await clearCachedCredentialFile();
        settings.setValue(scope, 'selectedAuthType', authMethod);
        // If the method changed TO Ollama, or was already Ollama and re-selected
        if (authMethod === AuthType.USE_OLLAMA) {
          // No immediate authFlow call here. The useEffect will pick up the change
          // in selectedAuthType and trigger the Ollama pre-check.
          // This also means onOllamaAuthTypeSelectedAndChecked will be called by that useEffect.
        }
      }
      setIsAuthDialogOpen(false);
      setAuthError(null); // Clear error when user makes a new selection or closes dialog
    },
    [settings, setAuthError], // Removed onOllamaAuthTypeSelectedAndChecked from here
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
