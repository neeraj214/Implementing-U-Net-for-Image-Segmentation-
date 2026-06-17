const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const segmentImage = async (file) => {
  try {
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
  } catch (error) {
    console.error('Error in segmentImage:', error);
    throw error;
  }
};

export const getMetrics = async () => {
  try {
    const response = await fetch(`${BASE_URL}/metrics`);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch metrics: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error in getMetrics:', error);
    throw error;
  }
};

export const checkHealth = async () => {
  try {
    const response = await fetch(`${BASE_URL}/health`);
    
    if (!response.ok) {
      throw new Error(`Failed to check health: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error in checkHealth:', error);
    throw error;
  }
};
