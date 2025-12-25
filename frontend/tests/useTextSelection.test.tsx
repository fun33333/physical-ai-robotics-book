/**
 * Unit tests for useTextSelection hook (T066 - Phase 6).
 *
 * Tests the text selection hook's ability to:
 * - Detect user text highlights
 * - Return selected text
 * - Handle multi-selection edge cases
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useTextSelection } from '../src/hooks/useTextSelection';

describe('useTextSelection', () => {
  let originalGetSelection: typeof window.getSelection;

  beforeEach(() => {
    originalGetSelection = window.getSelection;
  });

  afterEach(() => {
    window.getSelection = originalGetSelection;
    vi.restoreAllMocks();
  });

  describe('getSelectedText', () => {
    it('returns empty string when nothing is selected', () => {
      window.getSelection = vi.fn().mockReturnValue({
        toString: () => '',
        rangeCount: 0,
      });

      const { result } = renderHook(() => useTextSelection());

      expect(result.current.selectedText).toBe('');
    });

    it('returns selected text from DOM selection', () => {
      const selectedText = 'ROS 2 is a robotics framework';

      window.getSelection = vi.fn().mockReturnValue({
        toString: () => selectedText,
        rangeCount: 1,
      });

      const { result } = renderHook(() => useTextSelection());

      act(() => {
        // Simulate mouseup event
        window.dispatchEvent(new Event('mouseup'));
      });

      expect(result.current.selectedText).toBe(selectedText);
    });

    it('trims whitespace from selected text', () => {
      window.getSelection = vi.fn().mockReturnValue({
        toString: () => '  some text with spaces  ',
        rangeCount: 1,
      });

      const { result } = renderHook(() => useTextSelection());

      act(() => {
        window.dispatchEvent(new Event('mouseup'));
      });

      expect(result.current.selectedText).toBe('some text with spaces');
    });

    it('handles null selection gracefully', () => {
      window.getSelection = vi.fn().mockReturnValue(null);

      const { result } = renderHook(() => useTextSelection());

      expect(result.current.selectedText).toBe('');
    });
  });

  describe('clearSelection', () => {
    it('clears the selected text', () => {
      window.getSelection = vi.fn().mockReturnValue({
        toString: () => 'Selected text',
        rangeCount: 1,
        removeAllRanges: vi.fn(),
      });

      const { result } = renderHook(() => useTextSelection());

      act(() => {
        window.dispatchEvent(new Event('mouseup'));
      });

      expect(result.current.selectedText).toBe('Selected text');

      act(() => {
        result.current.clearSelection();
      });

      expect(result.current.selectedText).toBe('');
    });
  });

  describe('event listeners', () => {
    it('adds mouseup event listener on mount', () => {
      const addEventListenerSpy = vi.spyOn(window, 'addEventListener');

      renderHook(() => useTextSelection());

      expect(addEventListenerSpy).toHaveBeenCalledWith('mouseup', expect.any(Function));
    });

    it('removes mouseup event listener on unmount', () => {
      const removeEventListenerSpy = vi.spyOn(window, 'removeEventListener');

      const { unmount } = renderHook(() => useTextSelection());
      unmount();

      expect(removeEventListenerSpy).toHaveBeenCalledWith('mouseup', expect.any(Function));
    });

    it('updates selection on each mouseup event', () => {
      let callCount = 0;
      window.getSelection = vi.fn().mockImplementation(() => ({
        toString: () => `Selection ${++callCount}`,
        rangeCount: 1,
      }));

      const { result } = renderHook(() => useTextSelection());

      act(() => {
        window.dispatchEvent(new Event('mouseup'));
      });
      expect(result.current.selectedText).toBe('Selection 1');

      act(() => {
        window.dispatchEvent(new Event('mouseup'));
      });
      expect(result.current.selectedText).toBe('Selection 2');
    });
  });

  describe('multi-selection handling', () => {
    it('handles text selected across multiple elements', () => {
      window.getSelection = vi.fn().mockReturnValue({
        toString: () => 'Text from\nmultiple\nlines',
        rangeCount: 1,
      });

      const { result } = renderHook(() => useTextSelection());

      act(() => {
        window.dispatchEvent(new Event('mouseup'));
      });

      expect(result.current.selectedText).toBe('Text from\nmultiple\nlines');
    });

    it('handles very long selections by truncating', () => {
      const longText = 'A'.repeat(10000);
      window.getSelection = vi.fn().mockReturnValue({
        toString: () => longText,
        rangeCount: 1,
      });

      const { result } = renderHook(() => useTextSelection());

      act(() => {
        window.dispatchEvent(new Event('mouseup'));
      });

      // Should be truncated to max length (5000 chars per spec)
      expect(result.current.selectedText.length).toBeLessThanOrEqual(5000);
    });

    it('handles special characters in selection', () => {
      window.getSelection = vi.fn().mockReturnValue({
        toString: () => '<script>alert("xss")</script>',
        rangeCount: 1,
      });

      const { result } = renderHook(() => useTextSelection());

      act(() => {
        window.dispatchEvent(new Event('mouseup'));
      });

      // Should preserve text but not execute
      expect(result.current.selectedText).toBe('<script>alert("xss")</script>');
    });
  });

  describe('isSelecting state', () => {
    it('tracks selection state correctly', () => {
      window.getSelection = vi.fn().mockReturnValue({
        toString: () => 'Some text',
        rangeCount: 1,
      });

      const { result } = renderHook(() => useTextSelection());

      expect(result.current.hasSelection).toBe(false);

      act(() => {
        window.dispatchEvent(new Event('mouseup'));
      });

      expect(result.current.hasSelection).toBe(true);

      act(() => {
        result.current.clearSelection();
      });

      expect(result.current.hasSelection).toBe(false);
    });
  });
});
