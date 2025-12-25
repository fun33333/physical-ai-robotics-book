/**
 * Chat-related TypeScript interfaces for the RAG Chatbot.
 */

/**
 * Available tone options for responses.
 */
export type Tone = 'english' | 'roman_urdu' | 'bro_guide';

/**
 * User expertise level.
 */
export type UserLevel = 'beginner' | 'intermediate' | 'advanced';

/**
 * Source citation from retrieved chunks.
 */
export interface Source {
  chapter: string;
  section: string;
  confidence?: number;
}

/**
 * Request payload for the chat endpoint.
 */
export interface ChatRequest {
  query: string;
  selectedText?: string;
  tone?: Tone;
  userLevel?: UserLevel;
  conversationId?: string;
  userId?: string;
  conversationHistory?: ConversationEntry[];
}

/**
 * Response from the chat endpoint.
 */
export interface ChatResponse {
  response: string;
  sources: Source[];
  tone: Tone;
  sessionId: string;
  conversationId: string;
  conversationCount: number;
  validationStatus: 'approved' | 'flagged';
  latencyBreakdown: {
    retrieval?: number;
    generation?: number;
    formatting?: number;
    validation?: number;
  };
  totalLatencyMs: number;
}

/**
 * A single entry in conversation history.
 */
export interface ConversationEntry {
  query: string;
  response: string;
  tone?: Tone;
  createdAt?: string;
  sources?: Source[];
}

/**
 * Chat state for the component.
 */
export interface ChatState {
  messages: ConversationEntry[];
  isLoading: boolean;
  error: string | null;
  conversationId: string | null;
  selectedTone: Tone;
}

/**
 * Error response from the API.
 */
export interface ChatError {
  detail: string;
  status?: number;
}
