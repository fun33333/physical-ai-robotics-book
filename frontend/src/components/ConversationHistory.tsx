/**
 * ConversationHistory component for displaying chat messages.
 *
 * Renders the conversation as a list of question/answer pairs
 * with source citations and loading states.
 */

import React from 'react';
import type { ConversationEntry, Source } from '../types';

/**
 * Props for the ConversationHistory component.
 */
export interface ConversationHistoryProps {
  /** List of conversation entries to display */
  messages: ConversationEntry[];
  /** Whether a response is currently loading */
  isLoading?: boolean;
  /** Optional CSS class name */
  className?: string;
}

/**
 * Props for individual message display.
 */
interface MessageProps {
  entry: ConversationEntry;
  index: number;
}

/**
 * Render a single source citation.
 */
function SourceCitation({ source }: { source: Source }): React.ReactElement {
  return (
    <span className="source-citation">
      {source.chapter}
      {source.section && ` - ${source.section}`}
      {source.confidence !== undefined && (
        <span className="confidence"> ({Math.round(source.confidence * 100)}%)</span>
      )}
    </span>
  );
}

/**
 * Render a single message exchange (question + answer).
 */
function Message({ entry, index }: MessageProps): React.ReactElement {
  return (
    <div className="message-exchange" data-testid={`message-${index}`}>
      <div className="message user-message">
        <div className="message-role">You</div>
        <div className="message-content">{entry.query}</div>
      </div>
      <div className="message assistant-message">
        <div className="message-role">Assistant</div>
        <div className="message-content">{entry.response}</div>
        {entry.sources && entry.sources.length > 0 && (
          <div className="message-sources">
            <span className="sources-label">Sources: </span>
            {entry.sources.map((source, idx) => (
              <React.Fragment key={idx}>
                {idx > 0 && ', '}
                <SourceCitation source={source} />
              </React.Fragment>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * Loading indicator while waiting for response.
 */
function LoadingIndicator(): React.ReactElement {
  return (
    <div className="loading-indicator" aria-live="polite">
      <span className="loading-text">Thinking...</span>
    </div>
  );
}

/**
 * Component for displaying the full conversation history.
 */
export function ConversationHistory({
  messages,
  isLoading = false,
  className = '',
}: ConversationHistoryProps): React.ReactElement {
  return (
    <div className={`conversation-history ${className}`.trim()} role="log" aria-label="Conversation history">
      {messages.length === 0 && !isLoading && (
        <div className="empty-state">
          Ask a question about Physical AI and ROS 2 to get started!
        </div>
      )}
      {messages.map((entry, index) => (
        <Message key={index} entry={entry} index={index} />
      ))}
      {isLoading && <LoadingIndicator />}
    </div>
  );
}

export default ConversationHistory;
