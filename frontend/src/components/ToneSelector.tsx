/**
 * ToneSelector component for choosing response tone.
 *
 * Provides a dropdown to select between:
 * - English (formal)
 * - Roman Urdu (informal)
 * - Bro Guide (casual/friendly)
 */

import React from 'react';
import type { Tone } from '../types';

/**
 * Props for the ToneSelector component.
 */
export interface ToneSelectorProps {
  /** Currently selected tone */
  value: Tone;
  /** Callback when tone changes */
  onChange: (tone: Tone) => void;
  /** Whether the selector is disabled */
  disabled?: boolean;
  /** Optional CSS class name */
  className?: string;
}

/**
 * Tone option labels for display.
 */
const TONE_LABELS: Record<Tone, string> = {
  english: 'English (Formal)',
  roman_urdu: 'Roman Urdu',
  bro_guide: 'Bro Guide (Casual)',
};

/**
 * Dropdown component for selecting response tone.
 */
export function ToneSelector({
  value,
  onChange,
  disabled = false,
  className = '',
}: ToneSelectorProps): React.ReactElement {
  const handleChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    onChange(event.target.value as Tone);
  };

  return (
    <select
      value={value}
      onChange={handleChange}
      disabled={disabled}
      className={`tone-selector ${className}`.trim()}
      aria-label="Select response tone"
    >
      <option value="english">{TONE_LABELS.english}</option>
      <option value="roman_urdu">{TONE_LABELS.roman_urdu}</option>
      <option value="bro_guide">{TONE_LABELS.bro_guide}</option>
    </select>
  );
}

export default ToneSelector;
