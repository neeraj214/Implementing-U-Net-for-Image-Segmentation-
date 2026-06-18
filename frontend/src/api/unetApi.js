const BASE_URL = import.meta.env.VITE_API_URL || '/api';

export async function segmentImage(file) {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch(`${BASE_URL}/segment`, {
    method: 'POST',
    body: formData,
  });
  
  if (!response.ok) {
    throw new Error(`Failed to segment image: ${response.statusText}`);
  }
  
  return await response.json();
}

export async function getMetrics() {
  const response = await fetch(`${BASE_URL}/metrics`);
  
  if (!response.ok) {
    throw new Error(`Failed to fetch metrics: ${response.statusText}`);
  }
  
  return await response.json();
}

export async function checkHealth() {
  const response = await fetch(`${BASE_URL}/health`);
  
  if (!response.ok) {
    throw new Error(`Health check failed: ${response.statusText}`);
  }
  
  return await response.json();
}
