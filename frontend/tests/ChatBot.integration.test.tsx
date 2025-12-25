/**
 * Integration tests for ChatBot component (T068 - Phase 6).
 *
 * Tests the full ChatBot component integration:
 * - Mock API interaction
 * - Render component
 * - Type question and submit
 * - Verify response displays
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ChatBot } from '../src/components/ChatBot';
import { chatService } from '../src/services/chatService';

// Mock the chat service
vi.mock('../src/services/chatService', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../src/services/chatService')>();
  return {
    ...actual,
    chatService: {
      sendMessage: vi.fn(),
    },
  };
});

describe('ChatBot Integration', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('rendering', () => {
    it('renders the chat input field', () => {
      render(<ChatBot />);

      expect(screen.getByPlaceholderText(/ask a question/i)).toBeInTheDocument();
    });

    it('renders the send button', () => {
      render(<ChatBot />);

      expect(screen.getByRole('button', { name: /send/i })).toBeInTheDocument();
    });

    it('renders tone selector with default english', () => {
      render(<ChatBot />);

      expect(screen.getByRole('combobox')).toHaveValue('english');
    });
  });

  describe('user interaction', () => {
    it('allows typing in the input field', async () => {
      const user = userEvent.setup();
      render(<ChatBot />);

      const input = screen.getByPlaceholderText(/ask a question/i);
      await user.type(input, 'What is ROS 2?');

      expect(input).toHaveValue('What is ROS 2?');
    });

    it('submits question when send button is clicked', async () => {
      const mockResponse = {
        response: 'ROS 2 is a robotics framework',
        sources: [],
        tone: 'english',
        sessionId: 'sess-1',
        conversationId: 'sess-1',
        conversationCount: 1,
        validationStatus: 'approved',
        latencyBreakdown: {},
        totalLatencyMs: 500,
      };

      (chatService.sendMessage as ReturnType<typeof vi.fn>).mockResolvedValue(mockResponse);

      const user = userEvent.setup();
      render(<ChatBot />);

      const input = screen.getByPlaceholderText(/ask a question/i);
      await user.type(input, 'What is ROS 2?');

      const sendButton = screen.getByRole('button', { name: /send/i });
      await user.click(sendButton);

      expect(chatService.sendMessage).toHaveBeenCalledWith(
        expect.objectContaining({
          query: 'What is ROS 2?',
          tone: 'english',
        })
      );
    });

    it('submits question when Enter is pressed', async () => {
      const mockResponse = {
        response: 'ROS 2 is a robotics framework',
        sources: [],
        tone: 'english',
        sessionId: 'sess-1',
        conversationId: 'sess-1',
        conversationCount: 1,
        validationStatus: 'approved',
        latencyBreakdown: {},
        totalLatencyMs: 500,
      };

      (chatService.sendMessage as ReturnType<typeof vi.fn>).mockResolvedValue(mockResponse);

      const user = userEvent.setup();
      render(<ChatBot />);

      const input = screen.getByPlaceholderText(/ask a question/i);
      await user.type(input, 'What is ROS 2?{enter}');

      expect(chatService.sendMessage).toHaveBeenCalled();
    });

    it('clears input after successful submission', async () => {
      const mockResponse = {
        response: 'Answer',
        sources: [],
        tone: 'english',
        sessionId: 'sess-1',
        conversationId: 'sess-1',
        conversationCount: 1,
        validationStatus: 'approved',
        latencyBreakdown: {},
        totalLatencyMs: 500,
      };

      (chatService.sendMessage as ReturnType<typeof vi.fn>).mockResolvedValue(mockResponse);

      const user = userEvent.setup();
      render(<ChatBot />);

      const input = screen.getByPlaceholderText(/ask a question/i);
      await user.type(input, 'What is ROS 2?');
      await user.click(screen.getByRole('button', { name: /send/i }));

      await waitFor(() => {
        expect(input).toHaveValue('');
      });
    });
  });

  describe('response display', () => {
    it('displays the response after submission', async () => {
      const mockResponse = {
        response: 'ROS 2 is a flexible framework for writing robot software.',
        sources: [{ chapter: 'Module 1', section: 'Introduction' }],
        tone: 'english',
        sessionId: 'sess-1',
        conversationId: 'sess-1',
        conversationCount: 1,
        validationStatus: 'approved',
        latencyBreakdown: {},
        totalLatencyMs: 500,
      };

      (chatService.sendMessage as ReturnType<typeof vi.fn>).mockResolvedValue(mockResponse);

      const user = userEvent.setup();
      render(<ChatBot />);

      await user.type(screen.getByPlaceholderText(/ask a question/i), 'What is ROS 2?');
      await user.click(screen.getByRole('button', { name: /send/i }));

      await waitFor(() => {
        expect(screen.getByText(/ROS 2 is a flexible framework/)).toBeInTheDocument();
      });
    });

    it('displays sources when available', async () => {
      const mockResponse = {
        response: 'Answer with sources',
        sources: [{ chapter: 'Module 1', section: 'ROS Basics' }],
        tone: 'english',
        sessionId: 'sess-1',
        conversationId: 'sess-1',
        conversationCount: 1,
        validationStatus: 'approved',
        latencyBreakdown: {},
        totalLatencyMs: 500,
      };

      (chatService.sendMessage as ReturnType<typeof vi.fn>).mockResolvedValue(mockResponse);

      const user = userEvent.setup();
      render(<ChatBot />);

      await user.type(screen.getByPlaceholderText(/ask a question/i), 'Question');
      await user.click(screen.getByRole('button', { name: /send/i }));

      await waitFor(() => {
        expect(screen.getByText(/Module 1/)).toBeInTheDocument();
      });
    });
  });

  describe('loading state', () => {
    it('shows loading indicator while waiting for response', async () => {
      let resolvePromise: (value: unknown) => void;
      const promise = new Promise((resolve) => {
        resolvePromise = resolve;
      });

      (chatService.sendMessage as ReturnType<typeof vi.fn>).mockReturnValue(promise);

      const user = userEvent.setup();
      render(<ChatBot />);

      await user.type(screen.getByPlaceholderText(/ask a question/i), 'Question');
      await user.click(screen.getByRole('button', { name: /send/i }));

      expect(screen.getByText(/loading|thinking/i)).toBeInTheDocument();

      // Cleanup
      resolvePromise!({
        response: 'Answer',
        sources: [],
        tone: 'english',
        sessionId: 's',
        conversationId: 's',
        conversationCount: 1,
        validationStatus: 'approved',
        latencyBreakdown: {},
        totalLatencyMs: 100,
      });
    });

    it('disables send button while loading', async () => {
      let resolvePromise: (value: unknown) => void;
      const promise = new Promise((resolve) => {
        resolvePromise = resolve;
      });

      (chatService.sendMessage as ReturnType<typeof vi.fn>).mockReturnValue(promise);

      const user = userEvent.setup();
      render(<ChatBot />);

      await user.type(screen.getByPlaceholderText(/ask a question/i), 'Question');
      await user.click(screen.getByRole('button', { name: /send/i }));

      expect(screen.getByRole('button', { name: /send/i })).toBeDisabled();

      // Cleanup
      resolvePromise!({
        response: 'Answer',
        sources: [],
        tone: 'english',
        sessionId: 's',
        conversationId: 's',
        conversationCount: 1,
        validationStatus: 'approved',
        latencyBreakdown: {},
        totalLatencyMs: 100,
      });
    });
  });

  describe('error handling', () => {
    it('displays error message on API failure', async () => {
      (chatService.sendMessage as ReturnType<typeof vi.fn>).mockRejectedValue(
        new Error('API Error')
      );

      const user = userEvent.setup();
      render(<ChatBot />);

      await user.type(screen.getByPlaceholderText(/ask a question/i), 'Question');
      await user.click(screen.getByRole('button', { name: /send/i }));

      await waitFor(() => {
        expect(screen.getByText(/error|failed/i)).toBeInTheDocument();
      });
    });
  });

  describe('tone selection', () => {
    it('changes tone when selected', async () => {
      const mockResponse = {
        response: 'Bhai, ROS 2 ek robotics framework hai',
        sources: [],
        tone: 'bro_guide',
        sessionId: 'sess-1',
        conversationId: 'sess-1',
        conversationCount: 1,
        validationStatus: 'approved',
        latencyBreakdown: {},
        totalLatencyMs: 500,
      };

      (chatService.sendMessage as ReturnType<typeof vi.fn>).mockResolvedValue(mockResponse);

      const user = userEvent.setup();
      render(<ChatBot />);

      const toneSelector = screen.getByRole('combobox');
      await user.selectOptions(toneSelector, 'bro_guide');

      await user.type(screen.getByPlaceholderText(/ask a question/i), 'What is ROS?');
      await user.click(screen.getByRole('button', { name: /send/i }));

      expect(chatService.sendMessage).toHaveBeenCalledWith(
        expect.objectContaining({
          tone: 'bro_guide',
        })
      );
    });
  });

  describe('conversation history', () => {
    it('maintains conversation context across messages', async () => {
      const firstResponse = {
        response: 'First answer',
        sources: [],
        tone: 'english',
        sessionId: 'sess-1',
        conversationId: 'sess-1',
        conversationCount: 1,
        validationStatus: 'approved',
        latencyBreakdown: {},
        totalLatencyMs: 500,
      };

      const secondResponse = {
        response: 'Second answer',
        sources: [],
        tone: 'english',
        sessionId: 'sess-1',
        conversationId: 'sess-1',
        conversationCount: 2,
        validationStatus: 'approved',
        latencyBreakdown: {},
        totalLatencyMs: 500,
      };

      (chatService.sendMessage as ReturnType<typeof vi.fn>)
        .mockResolvedValueOnce(firstResponse)
        .mockResolvedValueOnce(secondResponse);

      const user = userEvent.setup();
      render(<ChatBot />);

      // First message
      await user.type(screen.getByPlaceholderText(/ask a question/i), 'First question');
      await user.click(screen.getByRole('button', { name: /send/i }));

      await waitFor(() => {
        expect(screen.getByText('First answer')).toBeInTheDocument();
      });

      // Second message
      await user.type(screen.getByPlaceholderText(/ask a question/i), 'Second question');
      await user.click(screen.getByRole('button', { name: /send/i }));

      // Second call should include conversationId
      expect(chatService.sendMessage).toHaveBeenLastCalledWith(
        expect.objectContaining({
          conversationId: 'sess-1',
        })
      );
    });
  });

  describe('selected text integration', () => {
    it('includes selected text in the request', async () => {
      const mockResponse = {
        response: 'Answer about selected text',
        sources: [],
        tone: 'english',
        sessionId: 'sess-1',
        conversationId: 'sess-1',
        conversationCount: 1,
        validationStatus: 'approved',
        latencyBreakdown: {},
        totalLatencyMs: 500,
      };

      (chatService.sendMessage as ReturnType<typeof vi.fn>).mockResolvedValue(mockResponse);

      const user = userEvent.setup();
      render(<ChatBot selectedText="ROS 2 middleware" />);

      await user.type(screen.getByPlaceholderText(/ask a question/i), 'What is this?');
      await user.click(screen.getByRole('button', { name: /send/i }));

      expect(chatService.sendMessage).toHaveBeenCalledWith(
        expect.objectContaining({
          selectedText: 'ROS 2 middleware',
        })
      );
    });
  });
});
