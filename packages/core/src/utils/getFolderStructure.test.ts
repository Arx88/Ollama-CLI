/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/* eslint-disable @typescript-eslint/no-explicit-any */
import { describe, it, expect, vi, beforeEach, afterEach, Mock } from 'vitest';
import fsPromises from 'fs/promises';
import * as fs from 'fs';
import { Dirent as FSDirent } from 'fs';
import * as nodePath from 'path';
import { getFolderStructure } from './getFolderStructure.js';
import * as gitUtils from './gitUtils.js';
import { FileDiscoveryService } from '../services/fileDiscoveryService.js';

vi.mock('path', async (importOriginal) => {
  const original = (await importOriginal()) as typeof nodePath;
  return {
    ...original,
    resolve: vi.fn((str) => str),
    // Other path functions (basename, join, normalize, etc.) will use original implementation
  };
});

vi.mock('fs/promises');
vi.mock('fs');
vi.mock('./gitUtils.js');

// Import 'path' again here, it will be the mocked version
import * as path from 'path';

// Removed the old createDirent helper.
// createDirentForTest will be the only helper.
const createDirentForTest = (
  name: string,
  type: 'file' | 'dir',
  parentDir: string,
): FSDirent => {
  const direntPath = nodePath.join(parentDir, name);
  return {
    name,
    isFile: () => type === 'file',
    isDirectory: () => type === 'dir',
    isBlockDevice: () => false,
    isCharacterDevice: () => false,
    isSymbolicLink: () => false,
    isFIFO: () => false,
    isSocket: () => false,
    // Add path and parentPath, and cast to satisfy the code expecting these,
    // even if @types/node doesn't fully reflect them for FSDirent yet.
    path: direntPath,
    parentPath: parentDir,
  } as FSDirent; // Cast to FSDirent. If errors persist, will use FSDirent & {path:string...}
};

describe('getFolderStructure', () => {
  let mockFsStructure: Record<string, FSDirent[]> = {};

  beforeEach(() => {
    vi.resetAllMocks();
    (path.resolve as Mock).mockImplementation((str: string) => str);
    (path.join as Mock).mockImplementation((...args: string[]) =>
      nodePath.join(...args),
    ); // Ensure join is mocked if path is mocked
    (path.normalize as Mock).mockImplementation((str: string) =>
      nodePath.normalize(str),
    );

    // Initialize mockFsStructure here because createDirentForTest needs parentPath
    mockFsStructure = {
      '/testroot': [
        createDirentForTest('file1.txt', 'file', '/testroot'),
        createDirentForTest('subfolderA', 'dir', '/testroot'),
        createDirentForTest('emptyFolder', 'dir', '/testroot'),
        createDirentForTest('.hiddenfile', 'file', '/testroot'),
        createDirentForTest('node_modules', 'dir', '/testroot'),
      ],
      '/testroot/subfolderA': [
        createDirentForTest('fileA1.ts', 'file', '/testroot/subfolderA'),
        createDirentForTest('fileA2.js', 'file', '/testroot/subfolderA'),
        createDirentForTest('subfolderB', 'dir', '/testroot/subfolderA'),
      ],
      '/testroot/subfolderA/subfolderB': [
        createDirentForTest(
          'fileB1.md',
          'file',
          '/testroot/subfolderA/subfolderB',
        ),
      ],
      '/testroot/emptyFolder': [],
      '/testroot/node_modules': [
        createDirentForTest('somepackage', 'dir', '/testroot/node_modules'),
      ],
      '/testroot/manyFilesFolder': Array.from({ length: 10 }, (_, i) =>
        createDirentForTest(
          `file-${i}.txt`,
          'file',
          '/testroot/manyFilesFolder',
        ),
      ),
      '/testroot/manyFolders': Array.from({ length: 5 }, (_, i) =>
        createDirentForTest(`folder-${i}`, 'dir', '/testroot/manyFolders'),
      ),
      ...Array.from({ length: 5 }, (_, i) => {
        const parent = `/testroot/manyFolders/folder-${i}`;
        return {
          [parent]: [createDirentForTest('child.txt', 'file', parent)],
        };
      }).reduce((acc, val) => ({ ...acc, ...val }), {}),
      '/testroot/deepFolders': [
        createDirentForTest('level1', 'dir', '/testroot/deepFolders'),
      ],
      '/testroot/deepFolders/level1': [
        createDirentForTest('level2', 'dir', '/testroot/deepFolders/level1'),
      ],
      '/testroot/deepFolders/level1/level2': [
        createDirentForTest(
          'level3',
          'dir',
          '/testroot/deepFolders/level1/level2',
        ),
      ],
      '/testroot/deepFolders/level1/level2/level3': [
        createDirentForTest(
          'file.txt',
          'file',
          '/testroot/deepFolders/level1/level2/level3',
        ),
      ],
    };

    (fsPromises.readdir as Mock).mockImplementation(
      async (dirPath: string | Buffer | URL) => {
        const normalizedPath = nodePath.normalize(dirPath.toString()); // Use actual nodePath
        if (mockFsStructure[normalizedPath]) {
          return mockFsStructure[normalizedPath];
        }
        throw Object.assign(
          new Error(
            `ENOENT: no such file or directory, scandir '${normalizedPath}'`,
          ),
          { code: 'ENOENT' },
        );
      },
    );
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('should return basic folder structure', async () => {
    const structure = await getFolderStructure('/testroot/subfolderA');
    const expected = `
Showing up to 200 items (files + folders).

/testroot/subfolderA/
├───fileA1.ts
├───fileA2.js
└───subfolderB/
    └───fileB1.md
`.trim();
    expect(structure.trim()).toBe(expected);
  });

  it('should handle an empty folder', async () => {
    const structure = await getFolderStructure('/testroot/emptyFolder');
    const expected = `
Showing up to 200 items (files + folders).

/testroot/emptyFolder/
`.trim();
    expect(structure.trim()).toBe(expected.trim());
  });

  it('should ignore folders specified in ignoredFolders (default)', async () => {
    const structure = await getFolderStructure('/testroot');
    const expected = `
Showing up to 200 items (files + folders). Folders or files indicated with ... contain more items not shown, were ignored, or the display limit (200 items) was reached.

/testroot/
├───.hiddenfile
├───file1.txt
├───emptyFolder/
├───node_modules/...
└───subfolderA/
    ├───fileA1.ts
    ├───fileA2.js
    └───subfolderB/
        └───fileB1.md
`.trim();
    expect(structure.trim()).toBe(expected);
  });

  it('should ignore folders specified in custom ignoredFolders', async () => {
    const structure = await getFolderStructure('/testroot', {
      ignoredFolders: new Set(['subfolderA', 'node_modules']),
    });
    const expected = `
Showing up to 200 items (files + folders). Folders or files indicated with ... contain more items not shown, were ignored, or the display limit (200 items) was reached.

/testroot/
├───.hiddenfile
├───file1.txt
├───emptyFolder/
├───node_modules/...
└───subfolderA/...
`.trim();
    expect(structure.trim()).toBe(expected);
  });

  it('should filter files by fileIncludePattern', async () => {
    const structure = await getFolderStructure('/testroot/subfolderA', {
      fileIncludePattern: /\.ts$/,
    });
    const expected = `
Showing up to 200 items (files + folders).

/testroot/subfolderA/
├───fileA1.ts
└───subfolderB/
`.trim();
    expect(structure.trim()).toBe(expected);
  });

  it('should handle maxItems truncation for files within a folder', async () => {
    const structure = await getFolderStructure('/testroot/subfolderA', {
      maxItems: 3,
    });
    const expected = `
Showing up to 3 items (files + folders).

/testroot/subfolderA/
├───fileA1.ts
├───fileA2.js
└───subfolderB/
`.trim();
    expect(structure.trim()).toBe(expected);
  });

  it('should handle maxItems truncation for subfolders', async () => {
    const structure = await getFolderStructure('/testroot/manyFolders', {
      maxItems: 4,
    });
    const expectedRevised = `
Showing up to 4 items (files + folders). Folders or files indicated with ... contain more items not shown, were ignored, or the display limit (4 items) was reached.

/testroot/manyFolders/
├───folder-0/
├───folder-1/
├───folder-2/
├───folder-3/
└───...
`.trim();
    expect(structure.trim()).toBe(expectedRevised);
  });

  it('should handle maxItems that only allows the root folder itself', async () => {
    const structure = await getFolderStructure('/testroot/subfolderA', {
      maxItems: 1,
    });
    const expectedRevisedMax1 = `
Showing up to 1 items (files + folders). Folders or files indicated with ... contain more items not shown, were ignored, or the display limit (1 items) was reached.

/testroot/subfolderA/
├───fileA1.ts
├───...
└───...
`.trim();
    expect(structure.trim()).toBe(expectedRevisedMax1);
  });

  it('should handle non-existent directory', async () => {
    // Temporarily make fsPromises.readdir throw ENOENT for this specific path
    const originalReaddir = fsPromises.readdir;
    (fsPromises.readdir as Mock).mockImplementation(
      async (p: string | Buffer | URL) => {
        if (p === '/nonexistent') {
          throw Object.assign(new Error('ENOENT'), { code: 'ENOENT' });
        }
        return originalReaddir(p);
      },
    );

    const structure = await getFolderStructure('/nonexistent');
    expect(structure).toContain(
      'Error: Could not read directory "/nonexistent"',
    );
  });

  it('should handle deep folder structure within limits', async () => {
    const structure = await getFolderStructure('/testroot/deepFolders', {
      maxItems: 10,
    });
    const expected = `
Showing up to 10 items (files + folders).

/testroot/deepFolders/
└───level1/
    └───level2/
        └───level3/
            └───file.txt
`.trim();
    expect(structure.trim()).toBe(expected);
  });

  it('should truncate deep folder structure if maxItems is small', async () => {
    const structure = await getFolderStructure('/testroot/deepFolders', {
      maxItems: 3,
    });
    const expected = `
Showing up to 3 items (files + folders).

/testroot/deepFolders/
└───level1/
    └───level2/
        └───level3/
`.trim();
    expect(structure.trim()).toBe(expected);
  });
});

describe('getFolderStructure gitignore', () => {
  beforeEach(() => {
    vi.resetAllMocks();
    (path.resolve as Mock).mockImplementation((str: string) => str);

    (fsPromises.readdir as Mock).mockImplementation(async (p) => {
      const dirPath = p.toString();
      if (dirPath === '/test/project') {
        return [
          createDirentForTest('file1.txt', 'file', dirPath),
          createDirentForTest('node_modules', 'dir', dirPath),
          createDirentForTest('ignored.txt', 'file', dirPath),
          createDirentForTest('.gemini', 'dir', dirPath),
        ] as any;
      }
      if (dirPath === '/test/project/node_modules') {
        return [createDirentForTest('some-package', 'dir', dirPath)] as any;
      }
      if (dirPath === '/test/project/.gemini') {
        return [
          createDirentForTest('config.yaml', 'file', dirPath),
          createDirentForTest('logs.json', 'file', dirPath),
        ] as any;
      }
      return [];
    });

    (fs.readFileSync as Mock).mockImplementation((p) => {
      const path = p.toString();
      if (path === '/test/project/.gitignore') {
        return 'ignored.txt\nnode_modules/\n.gemini/\n!/.gemini/config.yaml';
      }
      return '';
    });

    vi.mocked(gitUtils.isGitRepository).mockReturnValue(true);
  });

  it('should ignore files and folders specified in .gitignore', async () => {
    const fileService = new FileDiscoveryService('/test/project');
    const structure = await getFolderStructure('/test/project', {
      fileService,
    });
    expect(structure).not.toContain('ignored.txt');
    expect(structure).toContain('node_modules/...');
    expect(structure).not.toContain('logs.json');
  });

  it('should not ignore files if respectGitIgnore is false', async () => {
    const fileService = new FileDiscoveryService('/test/project');
    const structure = await getFolderStructure('/test/project', {
      fileService,
      respectGitIgnore: false,
    });
    expect(structure).toContain('ignored.txt');
    // node_modules is still ignored by default
    expect(structure).toContain('node_modules/...');
  });
});
