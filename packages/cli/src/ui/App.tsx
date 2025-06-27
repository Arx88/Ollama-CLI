/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { useCallback, useEffect, useMemo, useState, useRef } from 'react';
import {
  Box,
  DOMElement,
  measureElement,
  Static,
  Text,
  useStdin,
  useStdout,
  useInput,
  type Key as InkKeyType,
} from 'ink';
import { StreamingState, type HistoryItem, MessageType } from './types.js';
import { useTerminalSize } from './hooks/useTerminalSize.js';
import { useGeminiStream } from './hooks/useGeminiStream.js';
import { useLoadingIndicator } from './hooks/useLoadingIndicator.js';
import { useThemeCommand } from './hooks/useThemeCommand.js';
import { useAuthCommand } from './hooks/useAuthCommand.js';
import { useEditorSettings } from './hooks/useEditorSettings.js';
import { useSlashCommandProcessor } from './hooks/slashCommandProcessor.js';
import { useAutoAcceptIndicator } from './hooks/useAutoAcceptIndicator.js';
import { useConsoleMessages } from './hooks/useConsoleMessages.js';
import { Header } from './components/Header.js';
import { LoadingIndicator } from './components/LoadingIndicator.js';
import { AutoAcceptIndicator } from './components/AutoAcceptIndicator.js';
import { ShellModeIndicator } from './components/ShellModeIndicator.js';
import { InputPrompt } from './components/InputPrompt.js';
import { Footer } from './components/Footer.js';
import { ThemeDialog } from './components/ThemeDialog.js';
import { AuthDialog } from './components/AuthDialog.js';
import { AuthInProgress } from './components/AuthInProgress.js';
import { EditorSettingsDialog } from './components/EditorSettingsDialog.js';
import { OllamaModelDialog } from './components/OllamaModelDialog.js'; // Added
import { useOllamaModelSelection } from './hooks/useOllamaModelSelection.js'; // Added
import { Colors } from './colors.js';
import { Help } from './components/Help.js';
import { loadHierarchicalGeminiMemory } from '../config/config.js';
import { LoadedSettings, SettingScope } from '../config/settings.js'; // Added SettingScope
import { Tips } from './components/Tips.js';
import { useConsolePatcher } from './components/ConsolePatcher.js';
import { DetailedMessagesDisplay } from './components/DetailedMessagesDisplay.js';
import { HistoryItemDisplay } from './components/HistoryItemDisplay.js';
import { ContextSummaryDisplay } from './components/ContextSummaryDisplay.js';
import { useHistory } from './hooks/useHistoryManager.js';
import process from 'node:process';
import {
  getErrorMessage,
  type Config,
  getAllGeminiMdFilenames,
  ApprovalMode,
  isEditorAvailable,
  EditorType,
  AuthType,
  OllamaClient, // Added OllamaClient
} from '@google/gemini-cli-core';
import { validateAuthMethod } from '../config/auth.js';
import { useLogger } from './hooks/useLogger.js';
import { StreamingContext } from './contexts/StreamingContext.js';
import {
  SessionStatsProvider,
  useSessionStats,
} from './contexts/SessionContext.js';
import { useGitBranchName } from './hooks/useGitBranchName.js';
import { useTextBuffer } from './components/shared/text-buffer.js';
import * as fs from 'fs';
import { UpdateNotification } from './components/UpdateNotification.js';
import { checkForUpdates } from './utils/updateCheck.js';
import ansiEscapes from 'ansi-escapes';
import { OverflowProvider } from './contexts/OverflowContext.js';
import { ShowMoreLines } from './components/ShowMoreLines.js';
import { logToFile } from '../utils/fileLogger.js'; // Added for error logging

const CTRL_EXIT_PROMPT_DURATION_MS = 1000;

interface AppProps {
  config: Config;
  settings: LoadedSettings;
  startupWarnings?: string[];
}

export const AppWrapper = (props: AppProps) => (
  <SessionStatsProvider>
    <App {...props} />
  </SessionStatsProvider>
);

const App = ({ config, settings, startupWarnings = [] }: AppProps) => {
  const [updateMessage, setUpdateMessage] = useState<string | null>(null);
  const { stdout } = useStdout();

  useEffect(() => {
    checkForUpdates().then(setUpdateMessage);
  }, []);

  const { history, addItem, clearItems, loadHistory } = useHistory();
  const {
    consoleMessages,
    handleNewMessage,
    clearConsoleMessages: clearConsoleMessagesState,
  } = useConsoleMessages();
  const { stats: sessionStats } = useSessionStats();
  const [staticNeedsRefresh, setStaticNeedsRefresh] = useState(false);
  const [staticKey, setStaticKey] = useState(0);
  const refreshStatic = useCallback(() => {
    stdout.write(ansiEscapes.clearTerminal);
    setStaticKey((prev) => prev + 1);
  }, [setStaticKey, stdout]);

  const [geminiMdFileCount, setGeminiMdFileCount] = useState<number>(0);
  const [debugMessage, setDebugMessage] = useState<string>('');
  const [showHelp, setShowHelp] = useState<boolean>(false);
  const [themeError, setThemeError] = useState<string | null>(null);
  const [authError, setAuthError] = useState<string | null>(null);
  const [editorError, setEditorError] = useState<string | null>(null);
  const [footerHeight, setFooterHeight] = useState<number>(0);
  const [corgiMode, setCorgiMode] = useState(false);
  const [currentModel, setCurrentModel] = useState(config.getModel());
  const [shellModeActive, setShellModeActive] = useState(false);
  const [showErrorDetails, setShowErrorDetails] = useState<boolean>(false);
  const [showToolDescriptions, setShowToolDescriptions] =
    useState<boolean>(false);
  const [ctrlCPressedOnce, setCtrlCPressedOnce] = useState(false);
  const [quittingMessages, setQuittingMessages] = useState<
    HistoryItem[] | null
  >(null);
  const ctrlCTimerRef = useRef<NodeJS.Timeout | null>(null);
  const [ctrlDPressedOnce, setCtrlDPressedOnce] = useState(false);
  const ctrlDTimerRef = useRef<NodeJS.Timeout | null>(null);
  const [constrainHeight, setConstrainHeight] = useState<boolean>(true);

  const errorCount = useMemo(
    () => consoleMessages.filter((msg) => msg.type === 'error').length,
    [consoleMessages],
  );

  const {
    isThemeDialogOpen,
    openThemeDialog,
    handleThemeSelect,
    handleThemeHighlight,
  } = useThemeCommand(settings, setThemeError, addItem);

  const {
    isAuthDialogOpen,
    openAuthDialog,
    handleAuthSelect,
    handleAuthHighlight,
    isAuthenticating,
    cancelAuthentication,
  } = useAuthCommand(settings, setAuthError, config);

  useEffect(() => {
    if (settings.merged.selectedAuthType) {
      const error = validateAuthMethod(settings.merged.selectedAuthType);
      if (error) {
        setAuthError(error);
        openAuthDialog();
      }
    }
  }, [settings.merged.selectedAuthType, openAuthDialog, setAuthError]);

  const {
    isEditorDialogOpen,
    openEditorDialog,
    handleEditorSelect,
    exitEditorDialog,
  } = useEditorSettings(settings, setEditorError, addItem);

  // Ollama Model Selection Hook
  const {
    isOllamaModelDialogOpen,
    isLoadingModels: isLoadingOllamaModels,
    availableModels: availableOllamaModels,
    errorLoadingModels: errorLoadingOllamaModels,
    openOllamaModelDialog,
    // `onModelSelectedFromDialog` will be passed to OllamaModelDialog's onSelect prop
    // `handleDialogClose` will be passed to OllamaModelDialog's onCancel prop
    onModelSelectedFromDialog,
    handleDialogClose,
  } = useOllamaModelSelection(
    config,
    settings,
    // onModelSelected: Callback when a model is chosen from OllamaModelDialog
    async (selectedModelName: string) => {
      if (!config) return;
      addItem(
        {
          type: MessageType.INFO,
          text: `Configuring Ollama with model: ${selectedModelName}...`,
        },
        Date.now(),
      );
      try {
        settings.setValue(SettingScope.User, 'ollamaModel', selectedModelName); // Persist the choice
        config.setOllamaModel(selectedModelName); // Set in current session config
        await config.refreshAuth(AuthType.USE_OLLAMA); // Re-initialize client with new model
        addItem(
          {
            type: MessageType.INFO,
            text: `Ollama configured with ${selectedModelName}.`,
          },
          Date.now(),
        );
        setCurrentModel(config.getModel());
        refreshStatic();
        logToFile(
          `[App.tsx] Ollama successfully configured with model: ${selectedModelName}`,
        );
      } catch (e) {
        const errorMsg = getErrorMessage(e);
        addItem(
          {
            type: MessageType.ERROR,
            text: `Failed to apply Ollama model ${selectedModelName}: ${errorMsg}`,
          },
          Date.now(),
        );
        logToFile(
          `[App.tsx] Error applying Ollama model ${selectedModelName}: ${errorMsg}`,
        );
        setAuthError(
          `Failed to apply Ollama model ${selectedModelName}: ${errorMsg}. Please select auth method again.`,
        );
        settings.setValue(SettingScope.User, 'selectedAuthType', undefined);
        settings.setValue(SettingScope.User, 'ollamaModel', undefined);
        openAuthDialog(); // Guide user back to auth selection
      }
    },
    // onDialogCancelled: Callback when OllamaModelDialog is cancelled
    () => {
      addItem(
        { type: MessageType.INFO, text: 'Ollama model selection cancelled.' },
        Date.now(),
      );
      logToFile('[App.tsx] Ollama model selection was cancelled by user.');
      // If selection is cancelled, always clear Ollama settings and go back to AuthDialog
      // This ensures the user makes an active choice.
      setAuthError(
        'Ollama model selection was cancelled. Please choose an authentication method.',
      );
      settings.setValue(SettingScope.User, 'selectedAuthType', undefined);
      settings.setValue(SettingScope.User, 'ollamaModel', undefined);
      openAuthDialog(); // Guide user back to auth selection
    },
  );

  // Centralized Ollama Setup Logic
  const handleOllamaSetup = useCallback(async () => {
    logToFile('[App.tsx] Initiating Ollama setup check...');
    try {
      const ollamaClient = new OllamaClient(config);
      const availableModels = await ollamaClient.listModels();
      if (availableModels.models.length === 0) {
        logToFile('[App.tsx] Ollama setup: No models found.');
        setAuthError(
          'No Ollama models found. Ensure Ollama is running and models are installed, then type /auth.',
        );
        // Clear selected auth type so user is forced to re-select in AuthDialog
        settings.setValue(SettingScope.User, 'selectedAuthType', undefined);
        settings.setValue(SettingScope.User, 'ollamaModel', undefined); // Clear model too
        openAuthDialog();
        return;
      }
      logToFile(
        `[App.tsx] Ollama setup: Models found. Opening model selection dialog.`,
      );
      openOllamaModelDialog(); // Proceed to model selection
    } catch (e) {
      const errorMsg = getErrorMessage(e);
      logToFile(`[App.tsx] Ollama setup failed: ${errorMsg}`);
      setAuthError(
        `Ollama connection failed: ${errorMsg}. Ensure Ollama is running, then type /auth.`,
      );
      settings.setValue(SettingScope.User, 'selectedAuthType', undefined);
      settings.setValue(SettingScope.User, 'ollamaModel', undefined);
      openAuthDialog();
    }
  }, [config, openOllamaModelDialog, openAuthDialog, settings, setAuthError]);

  // Effect to handle Ollama setup when selected via AuthDialog
  const prevIsAuthDialogOpen = useRef(isAuthDialogOpen);
  useEffect(() => {
    // Check if AuthDialog was open and is now closed
    if (
      prevIsAuthDialogOpen.current &&
      !isAuthDialogOpen &&
      settings.merged.selectedAuthType === AuthType.USE_OLLAMA
    ) {
      logToFile(
        '[App.tsx] AuthDialog closed and Ollama is selected. Triggering Ollama setup.',
      );
      void handleOllamaSetup();
    }
    prevIsAuthDialogOpen.current = isAuthDialogOpen;
  }, [
    isAuthDialogOpen,
    settings.merged.selectedAuthType,
    handleOllamaSetup,
  ]);

  // Effect for initial app load logic for Ollama
  const initialLoadHandled = useRef(false);
  useEffect(() => {
    if (
      !initialLoadHandled.current &&
      !isAuthDialogOpen && // Don't run if auth dialog is already open for some reason
      settings.merged.selectedAuthType === AuthType.USE_OLLAMA
    ) {
      initialLoadHandled.current = true; // Mark as handled
      logToFile('[App.tsx] Initial load: Ollama is auth type.');
      if (!settings.merged.ollamaModel) {
        logToFile(
          '[App.tsx] Initial load: No Ollama model configured. Triggering setup.',
        );
        void handleOllamaSetup();
      } else {
        // Ollama is selected and a model is configured, try to verify and use it.
        logToFile(
          `[App.tsx] Initial load: Ollama model ${settings.merged.ollamaModel} is configured. Verifying...`,
        );
        const verifyAndUseOllama = async () => {
          try {
            const ollamaClient = new OllamaClient(config);
            const available = await ollamaClient.listModels(); // Simple check to see if Ollama is up
            if (available.models.length === 0) throw new Error("No models found in Ollama.");

            // Check if the specific model exists (optional, listModels might be enough for basic check)
            // For simplicity, we'll assume if Ollama is up, we can try to set the model.
            // A more robust check would be `ollamaClient.showModelDetails(settings.merged.ollamaModel)`.

            config.setOllamaModel(settings.merged.ollamaModel!); // Added non-null assertion
            await config.refreshAuth(AuthType.USE_OLLAMA);
            setCurrentModel(config.getModel()); // Update model display
            logToFile(
              `[App.tsx] Initial load: Successfully configured Ollama with ${settings.merged.ollamaModel}.`,
            );
          } catch (e) {
            const errorMsg = getErrorMessage(e);
            logToFile(
              `[App.tsx] Initial load: Failed to verify/use Ollama model ${settings.merged.ollamaModel}: ${errorMsg}`,
            );
            setAuthError(
              `Failed to initialize with Ollama model ${settings.merged.ollamaModel}: ${errorMsg}. Please re-authenticate via /auth.`,
            );
            settings.setValue(
              SettingScope.User,
              'selectedAuthType',
              undefined,
            );
            settings.setValue(SettingScope.User, 'ollamaModel', undefined);
            openAuthDialog();
          }
        };
        void verifyAndUseOllama();
      }
    }
    // Ensure initialLoadHandled.current is updated if auth type changes away from Ollama later
    // or if the auth dialog opens, so this effect can run again if needed.
    // However, for "initial load" it should strictly be once.
    // If settings change later, other useEffects should handle it.
  }, [
    settings.merged.selectedAuthType,
    settings.merged.ollamaModel,
    isAuthDialogOpen, // Re-evaluate if auth dialog opens
    handleOllamaSetup, // if setup needs to be re-triggered
    config,
    setCurrentModel,
    setAuthError,
    openAuthDialog,
    settings, // For setValue
  ]);


  const toggleCorgiMode = useCallback(() => {
    setCorgiMode((prev) => !prev);
  }, []);

  const performMemoryRefresh = useCallback(async () => {
    addItem(
      {
        type: MessageType.INFO,
        text: 'Refreshing hierarchical memory (GEMINI.md or other context files)...',
      },
      Date.now(),
    );
    try {
      const { memoryContent, fileCount } = await loadHierarchicalGeminiMemory(
        process.cwd(),
        config.getDebugMode(),
        config.getFileService(),
        config.getExtensionContextFilePaths(),
      );
      config.setUserMemory(memoryContent);
      config.setGeminiMdFileCount(fileCount);
      setGeminiMdFileCount(fileCount);

      addItem(
        {
          type: MessageType.INFO,
          text: `Memory refreshed successfully. ${memoryContent.length > 0 ? `Loaded ${memoryContent.length} characters from ${fileCount} file(s).` : 'No memory content found.'}`,
        },
        Date.now(),
      );
      if (config.getDebugMode()) {
        console.log(
          `[DEBUG] Refreshed memory content in config: ${memoryContent.substring(0, 200)}...`,
        );
      }
    } catch (error) {
      const errorMessage = getErrorMessage(error);
      addItem(
        {
          type: MessageType.ERROR,
          text: `Error refreshing memory: ${errorMessage}`,
        },
        Date.now(),
      );
      console.error('Error refreshing memory:', error);
    }
  }, [config, addItem]);

  // Watch for model changes (e.g., from Flash fallback)
  useEffect(() => {
    const checkModelChange = () => {
      const configModel = config.getModel();
      if (configModel !== currentModel) {
        setCurrentModel(configModel);
      }
    };

    // Check immediately and then periodically
    checkModelChange();
    const interval = setInterval(checkModelChange, 1000); // Check every second

    return () => clearInterval(interval);
  }, [config, currentModel]);

  // Set up Flash fallback handler
  useEffect(() => {
    const flashFallbackHandler = async (
      currentModel: string,
      fallbackModel: string,
    ): Promise<boolean> => {
      // Add message to UI history
      addItem(
        {
          type: MessageType.INFO,
          text: `⚡ Slow response times detected. Automatically switching from ${currentModel} to ${fallbackModel} for faster responses for the remainder of this session.
⚡ To avoid this you can utilize a Gemini API Key. See: https://goo.gle/gemini-cli-docs-auth#gemini-api-key
⚡ You can switch authentication methods by typing /auth`,
        },
        Date.now(),
      );
      return true; // Always accept the fallback
    };

    config.setFlashFallbackHandler(flashFallbackHandler);
  }, [config, addItem]);

  const {
    handleSlashCommand,
    slashCommands,
    pendingHistoryItems: pendingSlashCommandHistoryItems,
  } = useSlashCommandProcessor(
    config,
    settings,
    history,
    addItem,
    clearItems,
    loadHistory,
    refreshStatic,
    setShowHelp,
    setDebugMessage,
    openThemeDialog,
    openAuthDialog,
    openEditorDialog,
    performMemoryRefresh,
    toggleCorgiMode,
    showToolDescriptions,
    setQuittingMessages,
  );
  const pendingHistoryItems = [...pendingSlashCommandHistoryItems];

  const { rows: terminalHeight, columns: terminalWidth } = useTerminalSize();
  const isInitialMount = useRef(true);
  const { stdin, setRawMode } = useStdin();
  const isValidPath = useCallback((filePath: string): boolean => {
    try {
      return fs.existsSync(filePath) && fs.statSync(filePath).isFile();
    } catch (_e) {
      return false;
    }
  }, []);

  const widthFraction = 0.9;
  const inputWidth = Math.max(
    20,
    Math.floor(terminalWidth * widthFraction) - 3,
  );
  const suggestionsWidth = Math.max(60, Math.floor(terminalWidth * 0.8));

  const buffer = useTextBuffer({
    initialText: '',
    viewport: { height: 10, width: inputWidth },
    stdin,
    setRawMode,
    isValidPath,
  });

  const handleExit = useCallback(
    (
      pressedOnce: boolean,
      setPressedOnce: (value: boolean) => void,
      timerRef: React.MutableRefObject<NodeJS.Timeout | null>,
    ) => {
      if (pressedOnce) {
        if (timerRef.current) {
          clearTimeout(timerRef.current);
        }
        const quitCommand = slashCommands.find(
          (cmd) => cmd.name === 'quit' || cmd.altName === 'exit',
        );
        if (quitCommand) {
          quitCommand.action('quit', '', '');
        } else {
          process.exit(0);
        }
      } else {
        setPressedOnce(true);
        timerRef.current = setTimeout(() => {
          setPressedOnce(false);
          timerRef.current = null;
        }, CTRL_EXIT_PROMPT_DURATION_MS);
      }
    },
    [slashCommands],
  );

  useInput((input: string, key: InkKeyType) => {
    let enteringConstrainHeightMode = false;
    if (!constrainHeight) {
      // Automatically re-enter constrain height mode if the user types
      // anything. When constrainHeight==false, the user will experience
      // significant flickering so it is best to disable it immediately when
      // the user starts interacting with the app.
      enteringConstrainHeightMode = true;
      setConstrainHeight(true);

      // If our pending history item happens to exceed the terminal height we will most likely need to refresh
      // our static collection to ensure no duplication or tearing. This is currently working around a core bug
      // in Ink which we have a PR out to fix: https://github.com/vadimdemedes/ink/pull/717
      if (pendingHistoryItemRef.current && pendingHistoryItems.length > 0) {
        const pendingItemDimensions = measureElement(
          pendingHistoryItemRef.current,
        );
        if (pendingItemDimensions.height > availableTerminalHeight) {
          refreshStatic();
        }
      }
    }

    if (key.ctrl && input === 'o') {
      setShowErrorDetails((prev) => !prev);
    } else if (key.ctrl && input === 't') {
      const newValue = !showToolDescriptions;
      setShowToolDescriptions(newValue);

      const mcpServers = config.getMcpServers();
      if (Object.keys(mcpServers || {}).length > 0) {
        handleSlashCommand(newValue ? '/mcp desc' : '/mcp nodesc');
      }
    } else if (key.ctrl && (input === 'c' || input === 'C')) {
      handleExit(ctrlCPressedOnce, setCtrlCPressedOnce, ctrlCTimerRef);
    } else if (key.ctrl && (input === 'd' || input === 'D')) {
      if (buffer.text.length > 0) {
        // Do nothing if there is text in the input.
        return;
      }
      handleExit(ctrlDPressedOnce, setCtrlDPressedOnce, ctrlDTimerRef);
    } else if (key.ctrl && input === 's' && !enteringConstrainHeightMode) {
      setConstrainHeight(false);
    }
  });

  useConsolePatcher({
    onNewMessage: handleNewMessage,
    debugMode: config.getDebugMode(),
  });

  useEffect(() => {
    if (config) {
      setGeminiMdFileCount(config.getGeminiMdFileCount());
    }
  }, [config]);

  const getPreferredEditor = useCallback(() => {
    const editorType = settings.merged.preferredEditor;
    const isValidEditor = isEditorAvailable(editorType);
    if (!isValidEditor) {
      openEditorDialog();
      return;
    }
    return editorType as EditorType;
  }, [settings, openEditorDialog]);

  const onAuthError = useCallback(() => {
    setAuthError('reauth required');
    openAuthDialog();
  }, [openAuthDialog, setAuthError]);

  const {
    streamingState,
    submitQuery,
    initError,
    pendingHistoryItems: pendingGeminiHistoryItems,
    thought,
  } = useGeminiStream(
    config.getGeminiClient(),
    history,
    addItem,
    setShowHelp,
    config,
    setDebugMessage,
    handleSlashCommand,
    shellModeActive,
    getPreferredEditor,
    onAuthError,
    performMemoryRefresh,
  );
  pendingHistoryItems.push(...pendingGeminiHistoryItems);
  const { elapsedTime, currentLoadingPhrase } =
    useLoadingIndicator(streamingState);
  const showAutoAcceptIndicator = useAutoAcceptIndicator({ config });

  const handleFinalSubmit = useCallback(
    (submittedValue: string) => {
      const trimmedValue = submittedValue.trim();
      if (trimmedValue.length > 0) {
        submitQuery(trimmedValue);
      }
    },
    [submitQuery],
  );

  const logger = useLogger();
  const [userMessages, setUserMessages] = useState<string[]>([]);

  useEffect(() => {
    const fetchUserMessages = async () => {
      const pastMessagesRaw = (await logger?.getPreviousUserMessages()) || []; // Newest first

      const currentSessionUserMessages = history
        .filter(
          (item): item is HistoryItem & { type: 'user'; text: string } =>
            item.type === 'user' &&
            typeof item.text === 'string' &&
            item.text.trim() !== '',
        )
        .map((item) => item.text)
        .reverse(); // Newest first, to match pastMessagesRaw sorting

      // Combine, with current session messages being more recent
      const combinedMessages = [
        ...currentSessionUserMessages,
        ...pastMessagesRaw,
      ];

      // Deduplicate consecutive identical messages from the combined list (still newest first)
      const deduplicatedMessages: string[] = [];
      if (combinedMessages.length > 0) {
        deduplicatedMessages.push(combinedMessages[0]); // Add the newest one unconditionally
        for (let i = 1; i < combinedMessages.length; i++) {
          if (combinedMessages[i] !== combinedMessages[i - 1]) {
            deduplicatedMessages.push(combinedMessages[i]);
          }
        }
      }
      // Reverse to oldest first for useInputHistory
      setUserMessages(deduplicatedMessages.reverse());
    };
    fetchUserMessages();
  }, [history, logger]);

  const isInputActive = streamingState === StreamingState.Idle && !initError;

  const handleClearScreen = useCallback(() => {
    clearItems();
    clearConsoleMessagesState();
    console.clear();
    refreshStatic();
  }, [clearItems, clearConsoleMessagesState, refreshStatic]);

  const mainControlsRef = useRef<DOMElement>(null);
  const pendingHistoryItemRef = useRef<DOMElement>(null);

  useEffect(() => {
    if (mainControlsRef.current) {
      const fullFooterMeasurement = measureElement(mainControlsRef.current);
      setFooterHeight(fullFooterMeasurement.height);
    }
  }, [terminalHeight, consoleMessages, showErrorDetails]);

  const staticExtraHeight = /* margins and padding */ 3;
  const availableTerminalHeight = useMemo(
    () => terminalHeight - footerHeight - staticExtraHeight,
    [terminalHeight, footerHeight],
  );

  useEffect(() => {
    // skip refreshing Static during first mount
    if (isInitialMount.current) {
      isInitialMount.current = false;
      return;
    }

    // debounce so it doesn't fire up too often during resize
    const handler = setTimeout(() => {
      setStaticNeedsRefresh(false);
      refreshStatic();
    }, 300);

    return () => {
      clearTimeout(handler);
    };
  }, [terminalWidth, terminalHeight, refreshStatic]);

  useEffect(() => {
    if (!pendingHistoryItems.length) {
      return;
    }

    const pendingItemDimensions = measureElement(
      pendingHistoryItemRef.current!,
    );

    // If our pending history item happens to exceed the terminal height we will most likely need to refresh
    // our static collection to ensure no duplication or tearing. This is currently working around a core bug
    // in Ink which we have a PR out to fix: https://github.com/vadimdemedes/ink/pull/717
    if (pendingItemDimensions.height > availableTerminalHeight) {
      setStaticNeedsRefresh(true);
    }
  }, [pendingHistoryItems.length, availableTerminalHeight, streamingState]);

  useEffect(() => {
    if (streamingState === StreamingState.Idle && staticNeedsRefresh) {
      setStaticNeedsRefresh(false);
      refreshStatic();
    }
  }, [streamingState, refreshStatic, staticNeedsRefresh]);

  const filteredConsoleMessages = useMemo(() => {
    if (config.getDebugMode()) {
      return consoleMessages;
    }
    return consoleMessages.filter((msg) => msg.type !== 'debug');
  }, [consoleMessages, config]);

  const branchName = useGitBranchName(config.getTargetDir());

  const contextFileNames = useMemo(() => {
    const fromSettings = settings.merged.contextFileName;
    if (fromSettings) {
      return Array.isArray(fromSettings) ? fromSettings : [fromSettings];
    }
    return getAllGeminiMdFilenames();
  }, [settings.merged.contextFileName]);

  if (quittingMessages) {
    return (
      <Box flexDirection="column" marginBottom={1}>
        {quittingMessages.map((item) => (
          <HistoryItemDisplay
            key={item.id}
            availableTerminalHeight={
              constrainHeight ? availableTerminalHeight : undefined
            }
            terminalWidth={terminalWidth}
            item={item}
            isPending={false}
            config={config}
          />
        ))}
      </Box>
    );
  }
  const mainAreaWidth = Math.floor(terminalWidth * 0.9);
  const debugConsoleMaxHeight = Math.floor(Math.max(terminalHeight * 0.2, 5));
  // Arbitrary threshold to ensure that items in the static area are large
  // enough but not too large to make the terminal hard to use.
  const staticAreaMaxItemHeight = Math.max(terminalHeight * 4, 100);
  return (
    <StreamingContext.Provider value={streamingState}>
      <Box flexDirection="column" marginBottom={1} width="90%">
        {/*
         * The Static component is an Ink intrinsic in which there can only be 1 per application.
         * Because of this restriction we're hacking it slightly by having a 'header' item here to
         * ensure that it's statically rendered.
         *
         * Background on the Static Item: Anything in the Static component is written a single time
         * to the console. Think of it like doing a console.log and then never using ANSI codes to
         * clear that content ever again. Effectively it has a moving frame that every time new static
         * content is set it'll flush content to the terminal and move the area which it's "clearing"
         * down a notch. Without Static the area which gets erased and redrawn continuously grows.
         */}
        <Static
          key={staticKey}
          items={[
            <Box flexDirection="column" key="header">
              <Header terminalWidth={terminalWidth} />
              <Tips config={config} />
              {updateMessage && <UpdateNotification message={updateMessage} />}
            </Box>,
            ...history.map((h) => (
              <HistoryItemDisplay
                terminalWidth={mainAreaWidth}
                availableTerminalHeight={staticAreaMaxItemHeight}
                key={h.id}
                item={h}
                isPending={false}
                config={config}
              />
            )),
          ]}
        >
          {(item) => item}
        </Static>
        <OverflowProvider>
          <Box ref={pendingHistoryItemRef} flexDirection="column">
            {pendingHistoryItems.map((item, i) => (
              <HistoryItemDisplay
                key={i}
                availableTerminalHeight={
                  constrainHeight ? availableTerminalHeight : undefined
                }
                terminalWidth={mainAreaWidth}
                // TODO(taehykim): It seems like references to ids aren't necessary in
                // HistoryItemDisplay. Refactor later. Use a fake id for now.
                item={{ ...item, id: 0 }}
                isPending={true}
                config={config}
                isFocused={!isEditorDialogOpen}
              />
            ))}
            <ShowMoreLines constrainHeight={constrainHeight} />
          </Box>
        </OverflowProvider>

        {showHelp && <Help commands={slashCommands} />}

        <Box flexDirection="column" ref={mainControlsRef}>
          {startupWarnings.length > 0 && (
            <Box
              borderStyle="round"
              borderColor={Colors.AccentYellow}
              paddingX={1}
              marginY={1}
              flexDirection="column"
            >
              {startupWarnings.map((warning, index) => (
                <Text key={index} color={Colors.AccentYellow}>
                  {warning}
                </Text>
              ))}
            </Box>
          )}

          {isThemeDialogOpen ? (
            <Box flexDirection="column">
              {themeError && (
                <Box marginBottom={1}>
                  <Text color={Colors.AccentRed}>{themeError}</Text>
                </Box>
              )}
              <ThemeDialog
                onSelect={handleThemeSelect}
                onHighlight={handleThemeHighlight}
                settings={settings}
                availableTerminalHeight={
                  constrainHeight
                    ? terminalHeight - staticExtraHeight
                    : undefined
                }
                terminalWidth={mainAreaWidth}
              />
            </Box>
          ) : isAuthenticating ? (
            <AuthInProgress
              onTimeout={() => {
                setAuthError('Authentication timed out. Please try again.');
                cancelAuthentication();
                openAuthDialog();
              }}
            />
          ) : isAuthDialogOpen ? (
            <Box flexDirection="column">
              <AuthDialog
                onSelect={handleAuthSelect}
                onHighlight={handleAuthHighlight}
                settings={settings}
                initialErrorMessage={authError}
              />
            </Box>
          ) : isEditorDialogOpen ? (
            <Box flexDirection="column">
              {editorError && (
                <Box marginBottom={1}>
                  <Text color={Colors.AccentRed}>{editorError}</Text>
                </Box>
              )}
              <EditorSettingsDialog
                onSelect={handleEditorSelect}
                settings={settings}
                onExit={exitEditorDialog}
              />
            </Box>
          ) : isOllamaModelDialogOpen ? ( // Added condition for OllamaModelDialog
            <Box flexDirection="column">
              {isLoadingOllamaModels && <Text>Loading Ollama models...</Text>}
              {errorLoadingOllamaModels && (
                <Text color={Colors.AccentRed}>
                  Error: {errorLoadingOllamaModels}
                </Text>
              )}
              {!isLoadingOllamaModels && (
                <OllamaModelDialog
                  models={availableOllamaModels}
                  currentModel={settings.merged.ollamaModel}
                  onSelect={onModelSelectedFromDialog} // Changed from handleOllamaModelSelect
                  onCancel={handleDialogClose} // Changed from handleOllamaModelDialogClose
                />
              )}
            </Box>
          ) : (
            <>
              <LoadingIndicator
                thought={
                  streamingState === StreamingState.WaitingForConfirmation ||
                  config.getAccessibility()?.disableLoadingPhrases
                    ? undefined
                    : thought
                }
                currentLoadingPhrase={
                  config.getAccessibility()?.disableLoadingPhrases
                    ? undefined
                    : currentLoadingPhrase
                }
                elapsedTime={elapsedTime}
              />
              <Box
                marginTop={1}
                display="flex"
                justifyContent="space-between"
                width="100%"
              >
                <Box>
                  {process.env.GEMINI_SYSTEM_MD && (
                    <Text color={Colors.AccentRed}>|⌐■_■| </Text>
                  )}
                  {ctrlCPressedOnce ? (
                    <Text color={Colors.AccentYellow}>
                      Press Ctrl+C again to exit.
                    </Text>
                  ) : ctrlDPressedOnce ? (
                    <Text color={Colors.AccentYellow}>
                      Press Ctrl+D again to exit.
                    </Text>
                  ) : (
                    <ContextSummaryDisplay
                      geminiMdFileCount={geminiMdFileCount}
                      contextFileNames={contextFileNames}
                      mcpServers={config.getMcpServers()}
                      showToolDescriptions={showToolDescriptions}
                    />
                  )}
                </Box>
                <Box>
                  {showAutoAcceptIndicator !== ApprovalMode.DEFAULT &&
                    !shellModeActive && (
                      <AutoAcceptIndicator
                        approvalMode={showAutoAcceptIndicator}
                      />
                    )}
                  {shellModeActive && <ShellModeIndicator />}
                </Box>
              </Box>

              {showErrorDetails && (
                <OverflowProvider>
                  <DetailedMessagesDisplay
                    messages={filteredConsoleMessages}
                    maxHeight={
                      constrainHeight ? debugConsoleMaxHeight : undefined
                    }
                    width={inputWidth}
                  />
                  <ShowMoreLines constrainHeight={constrainHeight} />
                </OverflowProvider>
              )}

              {isInputActive && (
                <InputPrompt
                  buffer={buffer}
                  inputWidth={inputWidth}
                  suggestionsWidth={suggestionsWidth}
                  onSubmit={handleFinalSubmit}
                  userMessages={userMessages}
                  onClearScreen={handleClearScreen}
                  config={config}
                  slashCommands={slashCommands}
                  shellModeActive={shellModeActive}
                  setShellModeActive={setShellModeActive}
                />
              )}
            </>
          )}

          {initError && streamingState !== StreamingState.Responding && (
            <Box
              borderStyle="round"
              borderColor={Colors.AccentRed}
              paddingX={1}
              marginBottom={1}
            >
              {history.find(
                (item) =>
                  item.type === 'error' && item.text?.includes(initError),
              )?.text ? (
                <Text color={Colors.AccentRed}>
                  {
                    history.find(
                      (item) =>
                        item.type === 'error' && item.text?.includes(initError),
                    )?.text
                  }
                </Text>
              ) : (
                <>
                  <Text color={Colors.AccentRed}>
                    Initialization Error: {initError}
                  </Text>
                  <Text color={Colors.AccentRed}>
                    {' '}
                    Please check API key and configuration.
                  </Text>
                </>
              )}
            </Box>
          )}
          <Footer
            model={currentModel}
            targetDir={config.getTargetDir()}
            debugMode={config.getDebugMode()}
            branchName={branchName}
            debugMessage={debugMessage}
            corgiMode={corgiMode}
            errorCount={errorCount}
            showErrorDetails={showErrorDetails}
            showMemoryUsage={
              config.getDebugMode() || config.getShowMemoryUsage()
            }
            promptTokenCount={sessionStats.currentResponse.promptTokenCount}
            candidatesTokenCount={
              sessionStats.currentResponse.candidatesTokenCount
            }
            totalTokenCount={sessionStats.currentResponse.totalTokenCount}
          />
        </Box>
      </Box>
    </StreamingContext.Provider>
  );
};
