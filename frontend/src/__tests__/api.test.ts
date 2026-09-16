import { describe, it, expect } from 'vitest';
import { handleApiError, getApiBaseUrl } from '../services/api';

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

  describe('getApiBaseUrl Configuration', () => {
    it('should default to /api/v1 when VITE_API_BASE_URL is undefined or empty', () => {
      expect(getApiBaseUrl(undefined)).toBe('/api/v1');
      expect(getApiBaseUrl('')).toBe('/api/v1');
      expect(getApiBaseUrl('   ')).toBe('/api/v1');
    });

    it('should normalize bare domain production URLs (e.g. Render) to include /api/v1', () => {
      const renderDomain = 'https://datapilot-backend-vtke.onrender.com';
      expect(getApiBaseUrl(renderDomain)).toBe('https://datapilot-backend-vtke.onrender.com/api/v1');
    });

    it('should preserve explicit /api/v1 production URLs', () => {
      const renderFull = 'https://datapilot-backend-vtke.onrender.com/api/v1';
      expect(getApiBaseUrl(renderFull)).toBe('https://datapilot-backend-vtke.onrender.com/api/v1');
    });

    it('should handle /api suffix cleanly', () => {
      const renderApi = 'https://datapilot-backend-vtke.onrender.com/api';
      expect(getApiBaseUrl(renderApi)).toBe('https://datapilot-backend-vtke.onrender.com/api/v1');
    });

    it('should preserve local development URLs', () => {
      const localDev = 'http://localhost:8000/api/v1';
      expect(getApiBaseUrl(localDev)).toBe('http://localhost:8000/api/v1');
    });

    it('should trim trailing slashes correctly', () => {
      const trailingSlash = 'https://datapilot-backend-vtke.onrender.com/api/v1/';
      expect(getApiBaseUrl(trailingSlash)).toBe('https://datapilot-backend-vtke.onrender.com/api/v1');
    });
  });
});

