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
  // This function now only handles non-Ollama auth methods.
  // Ollama setup is initiated by App.tsx.
  // The `settings` parameter is kept for signature consistency but marked unused.
  // Guard against being called for Ollama, though the calling useEffect should prevent this.
  if (authMethod === AuthType.USE_OLLAMA) {
    console.warn(
      '[useAuthCommand] performAuthFlow called unexpectedly for Ollama. Flow control for Ollama is in App.tsx.',
    );
    return;
  }

  try {
    await config.refreshAuth(authMethod);
    console.log(`[useAuthCommand] Authenticated via "${authMethod}".`);
    setAuthError(null);
  } catch (e) {
    const errorMessage = getErrorMessage(e);
    console.error(
      `[useAuthCommand] Authentication flow failed for ${authMethod}:`,
      errorMessage,
    );
    setAuthError(`Authentication failed for ${authMethod}: ${errorMessage}.`);
    openAuthDialog(); // Re-open auth dialog on failure for non-Ollama methods
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
        setIsAuthenticating(false); // Ensure isAuthenticating is false if we exit early
        return;
      }

      // If Ollama is the selected type, this hook does nothing further for authentication.
      // App.tsx is responsible for handling the Ollama setup flow.
      // We just ensure isAuthenticating is false.
      if (settings.merged.selectedAuthType === AuthType.USE_OLLAMA) {
        console.log(
          '[useAuthCommand] Ollama selected via settings. App.tsx will handle setup. Ensuring isAuthenticating is false.',
        );
        if (isAuthenticating) {
          setIsAuthenticating(false);
        }
        return;
      }

      // For other auth types, proceed with authentication.
      try {
        setIsAuthenticating(true);
        await performAuthFlow(
          settings.merged.selectedAuthType as AuthType,
          config,
          settings, // Pass settings, though performAuthFlow might not use it.
          openAuthDialog,
          setAuthError,
        );
      } catch (e) {
        // This catch is primarily for unexpected errors if performAuthFlow itself throws
        // before it can set its own errors (though it should handle its own).
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
    settings.merged.selectedAuthType, // Specific dependency
    config,
    setAuthError,
    openAuthDialog,
    isAuthenticating, // Keep isAuthenticating to manage its state
  ]);

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
