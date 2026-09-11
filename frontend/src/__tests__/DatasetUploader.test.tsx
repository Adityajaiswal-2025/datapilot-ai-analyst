import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { DataPilotProvider } from '../context/DataPilotContext';
import { DatasetUploader } from '../components/DatasetUploader';

describe('DatasetUploader Component', () => {
  it('renders upload title and dropzone correctly', () => {
    render(
      <DataPilotProvider>
        <DatasetUploader />
      </DataPilotProvider>
    );

    expect(screen.getByText(/Dataset Ingestion Hub/i)).toBeDefined();
    expect(screen.getByText(/Drag & drop dataset file here/i)).toBeDefined();
  });
});
