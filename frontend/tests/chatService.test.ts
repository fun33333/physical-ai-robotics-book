/**
 * Unit tests for chat service client (T067 - Phase 6).
 *
 * Tests the chat service's ability to:
 * - Send POST /chat requests with correct payload
 * - Parse response JSON correctly
 * - Handle errors and retry on 429
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { chatService, ChatServiceError } from '../src/services/chatService';
import type { ChatRequest, ChatResponse } from '../src/types';

describe('chatService', () => {
  let originalFetch: typeof globalThis.fetch;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  describe('sendMessage', () => {
    const mockRequest: ChatRequest = {
      query: 'What is ROS 2?',
      selectedText: 'ROS 2 is a middleware',
      tone: 'english',
    };

    // API response is in snake_case
    const mockApiResponse = {
      response: 'ROS 2 is a robotics framework...',
      sources: [{ chapter: 'Module 1', section: 'Introduction' }],
      tone: 'english',
      session_id: 'session-123',
      conversation_id: 'session-123',
      conversation_count: 1,
      validation_status: 'approved',
      latency_breakdown: { retrieval: 100, generation: 500 },
      total_latency_ms: 600,
    };

    // Expected transformed response in camelCase
    const expectedResponse: ChatResponse = {
      response: 'ROS 2 is a robotics framework...',
      sources: [{ chapter: 'Module 1', section: 'Introduction' }],
      tone: 'english',
      sessionId: 'session-123',
      conversationId: 'session-123',
      conversationCount: 1,
      validationStatus: 'approved',
      latencyBreakdown: { retrieval: 100, generation: 500 },
      totalLatencyMs: 600,
    };

    it('sends POST request with correct payload', async () => {
      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockApiResponse),
      });

      await chatService.sendMessage(mockRequest);

      expect(globalThis.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/chat'),
        expect.objectContaining({
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            query: mockRequest.query,
            selected_text: mockRequest.selectedText,
            tone: mockRequest.tone,
          }),
        })
      );
    });

    it('parses response JSON correctly', async () => {
      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockApiResponse),
      });

      const result = await chatService.sendMessage(mockRequest);

      expect(result.response).toBe(expectedResponse.response);
      expect(result.sources).toEqual(expectedResponse.sources);
      expect(result.tone).toBe(expectedResponse.tone);
      expect(result.sessionId).toBe(expectedResponse.sessionId);
    });

    it('includes conversationId when provided', async () => {
      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockApiResponse),
      });

      await chatService.sendMessage({
        ...mockRequest,
        conversationId: 'existing-session',
      });

      expect(globalThis.fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          body: expect.stringContaining('existing-session'),
        })
      );
    });

    it('throws ChatServiceError on non-ok response', async () => {
      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        json: () => Promise.resolve({ detail: 'Internal server error' }),
      });

      await expect(chatService.sendMessage(mockRequest)).rejects.toThrow(ChatServiceError);
    });

    it('retries on 429 rate limit error', async () => {
      let callCount = 0;
      globalThis.fetch = vi.fn().mockImplementation(() => {
        callCount++;
        if (callCount === 1) {
          return Promise.resolve({
            ok: false,
            status: 429,
            json: () => Promise.resolve({ detail: 'Rate limit exceeded' }),
          });
        }
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(mockApiResponse),
        });
      });

      const result = await chatService.sendMessage(mockRequest);

      expect(callCount).toBe(2);
      expect(result.response).toBe(expectedResponse.response);
    });

    it('gives up after max retries on 429', async () => {
      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 429,
        json: () => Promise.resolve({ detail: 'Rate limit exceeded' }),
      });

      await expect(chatService.sendMessage(mockRequest)).rejects.toThrow();

      // Default max retries is 3
      expect(globalThis.fetch).toHaveBeenCalledTimes(3);
    });

    it('handles network errors gracefully', async () => {
      globalThis.fetch = vi.fn().mockRejectedValue(new Error('Network error'));

      await expect(chatService.sendMessage(mockRequest)).rejects.toThrow(ChatServiceError);
    });
  });

  describe('configuration', () => {
    it('uses default API URL when not configured', async () => {
      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({}),
      });

      await chatService.sendMessage({ query: 'test' });

      expect(globalThis.fetch).toHaveBeenCalledWith(
        expect.stringMatching(/\/chat$/),
        expect.any(Object)
      );
    });

    it('allows custom API URL configuration', async () => {
      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({}),
      });

      const customService = chatService.withBaseUrl('https://custom-api.example.com');
      await customService.sendMessage({ query: 'test' });

      expect(globalThis.fetch).toHaveBeenCalledWith(
        'https://custom-api.example.com/chat',
        expect.any(Object)
      );
    });
  });

  describe('request transformation', () => {
    it('converts camelCase to snake_case in request body', async () => {
      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({}),
      });

      await chatService.sendMessage({
        query: 'test',
        selectedText: 'some text',
        userLevel: 'beginner',
        conversationId: 'conv-123',
      });

      const callArgs = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0];
      const body = JSON.parse(callArgs[1].body);

      expect(body).toHaveProperty('selected_text');
      expect(body).toHaveProperty('user_level');
      expect(body).toHaveProperty('conversation_id');
    });

    it('omits undefined optional fields', async () => {
      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({}),
      });

      await chatService.sendMessage({ query: 'test' });

      const callArgs = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0];
      const body = JSON.parse(callArgs[1].body);

      expect(body).not.toHaveProperty('selected_text');
      expect(body).not.toHaveProperty('conversation_id');
    });
  });

  describe('response transformation', () => {
    it('converts snake_case to camelCase in response', async () => {
      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            response: 'answer',
            session_id: 'sess-1',
            conversation_id: 'sess-1',
            conversation_count: 5,
            validation_status: 'approved',
            latency_breakdown: { rag: 100 },
            total_latency_ms: 500,
            sources: [],
            tone: 'english',
          }),
      });

      const result = await chatService.sendMessage({ query: 'test' });

      expect(result.sessionId).toBe('sess-1');
      expect(result.conversationId).toBe('sess-1');
      expect(result.conversationCount).toBe(5);
      expect(result.validationStatus).toBe('approved');
      expect(result.latencyBreakdown).toEqual({ rag: 100 });
      expect(result.totalLatencyMs).toBe(500);
    });
  });
});
