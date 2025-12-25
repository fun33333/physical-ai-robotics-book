/**
 * useTextSelection hook for detecting text highlights.
 *
 * Provides functionality to:
 * - Detect user text highlights
 * - Return selected text
 * - Handle multi-selection edge cases
 */

import { useState, useEffect, useCallback } from 'react';

/** Maximum characters allowed for selected text */
const MAX_SELECTION_LENGTH = 5000;

/**
 * Hook return type for text selection.
 */
export interface UseTextSelectionReturn {
  /** The currently selected text */
  selectedText: string;
  /** Whether there is an active selection */
  hasSelection: boolean;
  /** Clear the current selection */
  clearSelection: () => void;
}

/**
 * Hook for detecting and managing text selection.
 *
 * Listens to mouseup events to capture text selections.
 * Automatically truncates very long selections.
 *
 * @returns Object containing selectedText, hasSelection state, and clearSelection function
 */
export function useTextSelection(): UseTextSelectionReturn {
  const [selectedText, setSelectedText] = useState<string>('');

  const handleSelection = useCallback(() => {
    const selection = window.getSelection();

    if (!selection) {
      setSelectedText('');
      return;
    }

    let text = selection.toString().trim();

    // Truncate very long selections
    if (text.length > MAX_SELECTION_LENGTH) {
      text = text.substring(0, MAX_SELECTION_LENGTH);
    }

    setSelectedText(text);
  }, []);

  const clearSelection = useCallback(() => {
    setSelectedText('');
    const selection = window.getSelection();
    if (selection && selection.removeAllRanges) {
      selection.removeAllRanges();
    }
  }, []);

  useEffect(() => {
    window.addEventListener('mouseup', handleSelection);

    return () => {
      window.removeEventListener('mouseup', handleSelection);
    };
  }, [handleSelection]);

  return {
    selectedText,
    hasSelection: selectedText.length > 0,
    clearSelection,
  };
}

export default useTextSelection;
