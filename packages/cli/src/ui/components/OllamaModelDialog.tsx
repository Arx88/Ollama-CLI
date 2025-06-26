/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { Box, Text, useInput } from 'ink';
import { RadioButtonSelect } from './shared/RadioButtonSelect.js'; // Assuming this path is correct
import { Colors } from '../colors.js';

interface OllamaModelDialogProps {
  models: string[];
  currentModel?: string; // Optional: to pre-select the currently configured model
  onSelect: (modelName: string) => void;
  onCancel: () => void; // To handle Esc key or explicit cancel action
}

export function OllamaModelDialog({
  models,
  currentModel,
  onSelect,
  onCancel,
}: OllamaModelDialogProps): React.JSX.Element {
  useInput((_input, key) => {
    if (key.escape) {
      onCancel();
    }
  });

  if (!models || models.length === 0) {
    return (
      <Box
        borderStyle="round"
        borderColor={Colors.AccentYellow}
        flexDirection="column"
        padding={1}
        width="100%"
      >
        <Text bold color={Colors.AccentYellow}>
          No Ollama Models Found
        </Text>
        <Text>
          Could not retrieve models from Ollama, or no models are installed.
        </Text>
        <Text>Please ensure Ollama is running and models are pulled.</Text>
        <Text> </Text>
        <Text color={Colors.Gray}>(Press Esc to close)</Text>
      </Box>
    );
  }

  const items = models.map((model) => ({
    label: model,
    value: model,
  }));

  let initialIndex = 0;
  if (currentModel) {
    const idx = models.indexOf(currentModel);
    if (idx !== -1) {
      initialIndex = idx;
    }
  }

  return (
    <Box
      borderStyle="round"
      borderColor={Colors.Gray}
      flexDirection="column"
      padding={1}
      width="100%"
    >
      <Text bold>Select Ollama Model</Text>
      <RadioButtonSelect
        items={items}
        initialIndex={initialIndex}
        onSelect={(item) => onSelect(item.value)}
        onCancel={onCancel} // Pass cancel if RadioButtonSelect supports it, or rely on useInput above
        isFocused={true}
      />
      <Box marginTop={1}>
        <Text color={Colors.Gray}>(Use Enter to select, Esc to cancel)</Text>
      </Box>
    </Box>
  );
}
