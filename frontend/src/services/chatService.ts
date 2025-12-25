/**
 * Chat service client for RAG Chatbot API.
 *
 * Provides:
 * - HTTP client for POST /chat
 * - Error handling
 * - Retry logic on 429 (quota exceeded)
 */

import type { ChatRequest, ChatResponse, Tone, UserLevel } from '../types';

/** Default API base URL */
const DEFAULT_BASE_URL = '/api';

/** Maximum retry attempts for rate limiting */
const MAX_RETRIES = 3;

/** Delay between retries in ms */
const RETRY_DELAY = 1000;

/**
 * Custom error class for chat service errors.
 */
export class ChatServiceError extends Error {
  public readonly status?: number;
  public readonly detail?: string;

  constructor(message: string, status?: number, detail?: string) {
    super(message);
    this.name = 'ChatServiceError';
    this.status = status;
    this.detail = detail;
  }
}

/**
 * API request body format (snake_case).
 */
interface ApiRequestBody {
  query: string;
  selected_text?: string;
  tone?: Tone;
  user_level?: UserLevel;
  conversation_id?: string;
  user_id?: string;
  conversation_history?: Array<{
    query: string;
    response: string;
    tone?: Tone;
    created_at?: string;
  }>;
}

/**
 * API response format (snake_case).
 */
interface ApiResponse {
  response: string;
  sources: Array<{ chapter: string; section: string; confidence?: number }>;
  tone: Tone;
  session_id: string;
  conversation_id: string;
  conversation_count: number;
  validation_status: 'approved' | 'flagged';
  latency_breakdown: Record<string, number>;
  total_latency_ms: number;
}

/**
 * Convert ChatRequest to API request body.
 */
function toApiRequestBody(request: ChatRequest): ApiRequestBody {
  const body: ApiRequestBody = {
    query: request.query,
  };

  if (request.selectedText !== undefined) {
    body.selected_text = request.selectedText;
  }
  if (request.tone !== undefined) {
    body.tone = request.tone;
  }
  if (request.userLevel !== undefined) {
    body.user_level = request.userLevel;
  }
  if (request.conversationId !== undefined) {
    body.conversation_id = request.conversationId;
  }
  if (request.userId !== undefined) {
    body.user_id = request.userId;
  }
  if (request.conversationHistory !== undefined) {
    body.conversation_history = request.conversationHistory.map((entry) => ({
      query: entry.query,
      response: entry.response,
      tone: entry.tone,
      created_at: entry.createdAt,
    }));
  }

  return body;
}

/**
 * Convert API response to ChatResponse.
 */
function fromApiResponse(response: ApiResponse): ChatResponse {
  return {
    response: response.response,
    sources: response.sources,
    tone: response.tone,
    sessionId: response.session_id,
    conversationId: response.conversation_id,
    conversationCount: response.conversation_count,
    validationStatus: response.validation_status,
    latencyBreakdown: response.latency_breakdown,
    totalLatencyMs: response.total_latency_ms,
  };
}

/**
 * Sleep for the specified duration.
 */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Chat service implementation.
 */
class ChatServiceImpl {
  private baseUrl: string;

  constructor(baseUrl: string = DEFAULT_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  /**
   * Create a new service instance with a custom base URL.
   */
  withBaseUrl(baseUrl: string): ChatServiceImpl {
    return new ChatServiceImpl(baseUrl);
  }

  /**
   * Send a message to the chat API.
   *
   * Handles rate limiting with automatic retry.
   *
   * @param request - The chat request
   * @returns The chat response
   * @throws ChatServiceError on failure
   */
  async sendMessage(request: ChatRequest): Promise<ChatResponse> {
    const url = `${this.baseUrl}/chat`;
    const body = toApiRequestBody(request);

    let lastError: Error | null = null;

    for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
      try {
        const response = await fetch(url, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(body),
        });

        if (response.ok) {
          const data: ApiResponse = await response.json();
          return fromApiResponse(data);
        }

        // Handle rate limiting
        if (response.status === 429) {
          if (attempt < MAX_RETRIES - 1) {
            await sleep(RETRY_DELAY * (attempt + 1));
            continue;
          }
        }

        // Handle other errors
        const errorData = await response.json().catch(() => ({}));
        throw new ChatServiceError(
          `Request failed with status ${response.status}`,
          response.status,
          errorData.detail
        );
      } catch (error) {
        if (error instanceof ChatServiceError) {
          throw error;
        }
        lastError = error as Error;
      }
    }

    throw new ChatServiceError(
      lastError?.message || 'Request failed after retries',
      undefined,
      lastError?.message
    );
  }
}

/**
 * Default chat service instance.
 */
export const chatService = new ChatServiceImpl();
