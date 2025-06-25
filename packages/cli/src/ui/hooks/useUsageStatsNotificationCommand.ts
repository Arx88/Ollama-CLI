/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useCallback } from 'react';
import { LoadedSettings, SettingScope } from '../../config/settings.js';

interface UseUsageStatsNotificationCommandReturn {
  isUsageStatsNotificationDialogOpen: boolean;
  openUsageStatsNotificationDialog: () => void;
  handleUsageStatsNotificationOptionSelect: (
    optionName: string | undefined,
    scope: SettingScope,
  ) => void;
}

export const useUsageStatsNotificationCommand = (
  loadedSettings: LoadedSettings,
): UseUsageStatsNotificationCommandReturn => {
  const usageStatsEnabled = loadedSettings.merged.usageStatisticsEnabled;

  // Initial state: Open dialog if no theme is set in either user or workspace settings
  const [isUsageStatsNotificationDialogOpen, setIsUsageStatsNotificationDialogOpen] = useState(
    usageStatsEnabled === undefined,
  );

  const openUsageStatsNotificationDialog = useCallback(() => {
    setIsUsageStatsNotificationDialogOpen(true);
  }, []);

  const handleUsageStatsNotificationOptionSelect = useCallback(
    (optionName: string | undefined, scope: SettingScope) => {
      try {
        loadedSettings.setValue(scope, 'theme', optionName); // Update the merged settings
      } finally {
        setIsUsageStatsNotificationDialogOpen(false); // Close the dialog
      }
    },
    [loadedSettings],
  );

  return {
    isUsageStatsNotificationDialogOpen,
    openUsageStatsNotificationDialog,
    handleUsageStatsNotificationOptionSelect,
  };
};
