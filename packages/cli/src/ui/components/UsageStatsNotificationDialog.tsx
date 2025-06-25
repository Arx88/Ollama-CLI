/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { Box, Text } from 'ink';
import { RadioButtonSelect, type RadioSelectItem } from './shared/RadioButtonSelect.js';
import { Colors } from '../colors.js';
import { LoadedSettings, SettingScope } from '../../config/settings.js';

interface UsageStatsNotificationDialogProps {
    onSelect: (value: string, scope: SettingScope) => void;
    settings: LoadedSettings;
}

const DIALOG_CHOICES: RadioSelectItem<string>[] = [
    {
        label: 'Acknowledge',
        value: 'acknowledge',
    },
    {
        label: 'Disable usage statistics',
        value: 'disable',
    },
];

export function UsageStatsNotificationDialog({ onSelect, settings }: UsageStatsNotificationDialogProps): React.JSX.Element {
    const [selectedScope, setSelectedScope] = useState<SettingScope>(
        SettingScope.User,
      );

    const handleOptionSelect = (optionName: string) => {
        onSelect(optionName, selectedScope);
    };

    return (
        <Box borderStyle="round"
          borderColor={Colors.Gray}
          flexDirection="column"
          padding={1}
          width="100%">
            <Text>
                To help improve Gemini CLI, we collect anonymized usage statistics.
            </Text>
            <Text>
                You can change this setting at any time by invoking /toggleUsageStatistics within Gemini CLI 
                or by changing the value of "usageStatisticsEnabled" in your ~/.gemini/settings.json file.
            </Text>
            <RadioButtonSelect
                items={DIALOG_CHOICES}
                onSelect={handleOptionSelect}
            />
        </Box>
    );
}