/**
 * RAG Chatbot Frontend - Main Entry Point
 *
 * Exports all public components, hooks, and services for use
 * in the Docusaurus integration.
 */

// Components
export { ChatBot, ConversationHistory, ToneSelector } from './components';
export type { ChatBotProps, ConversationHistoryProps, ToneSelectorProps } from './components';

// Hooks
export { useTextSelection } from './hooks';
export type { UseTextSelectionReturn } from './hooks';

// Services
export { chatService, ChatServiceError } from './services';

// Types
export type {
  Tone,
  UserLevel,
  Source,
  ChatRequest,
  ChatResponse,
  ConversationEntry,
  ChatState,
  ChatError,
} from './types';
