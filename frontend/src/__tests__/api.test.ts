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

  it('should return production-neutral message when API server is unreachable without exposing localhost', () => {
    const axiosUnreachableError = {
      isAxiosError: true,
      request: {},
      message: 'Network Error',
    };
    const msg = handleApiError(axiosUnreachableError, 'Fallback message');
    expect(msg).toBe('Unable to reach the DataPilot API server. Please check your connection and try again.');
    expect(msg).not.toContain('127.0.0.1');
    expect(msg).not.toContain('localhost');
  });
});

