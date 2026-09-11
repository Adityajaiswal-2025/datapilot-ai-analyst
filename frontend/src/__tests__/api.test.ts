import { describe, it, expect } from 'vitest';
import { handleApiError } from '../services/api';

describe('API Service Layer', () => {
  it('should format standard string error messages correctly', () => {
    const error = new Error('Network error');
    const msg = handleApiError(error, 'Fallback message');
    expect(msg).toBe('Network error');
  });

  it('should return fallback message for unknown error types', () => {
    const msg = handleApiError(null, 'Fallback message');
    expect(msg).toBe('Fallback message');
  });
});
