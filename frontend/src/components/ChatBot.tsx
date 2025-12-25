/**
 * ChatBot component - main chat interface for RAG Chatbot.
 *
 * Features:
 * - Text input for questions
 * - Tone selector for response style
 * - Conversation history display
 * - Loading and error states
 * - Multi-turn conversation support
 * - Selected text context support
 */

import React, { useState, useCallback, useRef, useEffect } from 'react';
import { chatService, ChatServiceError } from '../services/chatService';
import { ToneSelector } from './ToneSelector';
import { ConversationHistory } from './ConversationHistory';
import type { Tone, ConversationEntry, ChatState } from '../types';

/**
 * Props for the ChatBot component.
 */
export interface ChatBotProps {
  /** Text selected from the page to provide context */
  selectedText?: string;
  /** Initial tone setting */
  initialTone?: Tone;
  /** Optional CSS class name */
  className?: string;
  /** Callback when conversation starts */
  onConversationStart?: (conversationId: string) => void;
}

/**
 * Initial state for the chat component.
 */
const initialState: ChatState = {
  messages: [],
  isLoading: false,
  error: null,
  conversationId: null,
  selectedTone: 'english',
};

/**
 * Main ChatBot component providing the chat interface.
 */
export function ChatBot({
  selectedText,
  initialTone = 'english',
  className = '',
  onConversationStart,
}: ChatBotProps): React.ReactElement {
  const [state, setState] = useState<ChatState>({
    ...initialState,
    selectedTone: initialTone,
  });
  const [inputValue, setInputValue] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);
  const historyRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    if (historyRef.current) {
      historyRef.current.scrollTop = historyRef.current.scrollHeight;
    }
  }, [state.messages, state.isLoading]);

  /**
   * Handle tone change.
   */
  const handleToneChange = useCallback((tone: Tone) => {
    setState((prev) => ({ ...prev, selectedTone: tone }));
  }, []);

  /**
   * Handle input change.
   */
  const handleInputChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(event.target.value);
  }, []);

  /**
   * Submit a message to the chat API.
   */
  const handleSubmit = useCallback(
    async (event?: React.FormEvent) => {
      if (event) {
        event.preventDefault();
      }

      const query = inputValue.trim();
      if (!query || state.isLoading) {
        return;
      }

      // Clear input immediately
      setInputValue('');

      // Set loading state and add user message optimistically
      setState((prev) => ({
        ...prev,
        isLoading: true,
        error: null,
      }));

      try {
        const response = await chatService.sendMessage({
          query,
          selectedText,
          tone: state.selectedTone,
          conversationId: state.conversationId || undefined,
        });

        // Create new conversation entry
        const newEntry: ConversationEntry = {
          query,
          response: response.response,
          tone: response.tone,
          sources: response.sources,
        };

        // Update state with response
        setState((prev) => {
          const isNewConversation = !prev.conversationId;

          // Notify about new conversation
          if (isNewConversation && onConversationStart) {
            onConversationStart(response.conversationId);
          }

          return {
            ...prev,
            messages: [...prev.messages, newEntry],
            isLoading: false,
            error: null,
            conversationId: response.conversationId,
          };
        });
      } catch (error) {
        const errorMessage =
          error instanceof ChatServiceError
            ? error.detail || error.message
            : 'Failed to send message. Please try again.';

        setState((prev) => ({
          ...prev,
          isLoading: false,
          error: errorMessage,
        }));
      }
    },
    [inputValue, state.isLoading, state.selectedTone, state.conversationId, selectedText, onConversationStart]
  );

  /**
   * Handle Enter key press.
   */
  const handleKeyPress = useCallback(
    (event: React.KeyboardEvent<HTMLInputElement>) => {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit]
  );

  /**
   * Clear error message.
   */
  const clearError = useCallback(() => {
    setState((prev) => ({ ...prev, error: null }));
  }, []);

  return (
    <div className={`chatbot ${className}`.trim()}>
      <div className="chatbot-header">
        <h3 className="chatbot-title">Physical AI Assistant</h3>
        <ToneSelector
          value={state.selectedTone}
          onChange={handleToneChange}
          disabled={state.isLoading}
        />
      </div>

      <div className="chatbot-history" ref={historyRef}>
        <ConversationHistory
          messages={state.messages}
          isLoading={state.isLoading}
        />
      </div>

      {state.error && (
        <div className="chatbot-error" role="alert">
          <span className="error-message">{state.error}</span>
          <button
            type="button"
            className="error-dismiss"
            onClick={clearError}
            aria-label="Dismiss error"
          >
            Dismiss
          </button>
        </div>
      )}

      {selectedText && (
        <div className="chatbot-context">
          <span className="context-label">Context: </span>
          <span className="context-text">
            {selectedText.length > 100
              ? `${selectedText.substring(0, 100)}...`
              : selectedText}
          </span>
        </div>
      )}

      <form className="chatbot-input-form" onSubmit={handleSubmit}>
        <input
          ref={inputRef}
          type="text"
          className="chatbot-input"
          value={inputValue}
          onChange={handleInputChange}
          onKeyPress={handleKeyPress}
          placeholder="Ask a question about Physical AI..."
          disabled={state.isLoading}
          aria-label="Ask a question"
        />
        <button
          type="submit"
          className="chatbot-send-button"
          disabled={state.isLoading || !inputValue.trim()}
          aria-label="Send"
        >
          Send
        </button>
      </form>
    </div>
  );
}

export default ChatBot;
