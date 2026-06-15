export const predictImage = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await fetch('/api/predict', {
    method: 'POST',
    body: formData
  });
  if (!response.ok) {
    throw new Error('Prediction failed');
  }
  return await response.blob();
};
