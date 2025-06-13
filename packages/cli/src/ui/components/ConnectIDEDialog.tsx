/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Box, Text, useInput } from 'ink';
import { useCallback } from 'react';
import { Colors } from '../colors.js';

interface ConnectIDEDialogProps {
  onExit: () => void;
}

export const ConnectIDEDialog = ({ onExit }: ConnectIDEDialogProps) => {
  useInput(
    useCallback(() => {
      onExit();
    }, [onExit]),
  );

  return (
    <Box
      borderStyle="round"
      borderColor={Colors.AccentBlue}
      paddingX={1}
      flexDirection="column"
    >
      <Box marginBottom={1}>
        <Text bold>Connect IDE</Text>
      </Box>
      <Text>Press any key to exit.</Text>
    </Box>
  );
};
