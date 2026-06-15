import { useState } from 'react';
import { predictImage } from '../api/client';

const ImageUploader = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [predictionUrl, setPredictionUrl] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setImageUrl(URL.createObjectURL(file));
      setPredictionUrl(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedFile) return;
    setLoading(true);
    try {
      const blob = await predictImage(selectedFile);
      const url = URL.createObjectURL(blob);
      setPredictionUrl(url);
    } catch (error) {
      console.error('Error predicting image:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '20px', maxWidth: '800px', margin: '0 auto' }}>
      <h1 style={{ textAlign: 'center', marginBottom: '20px' }}>
        U-Net Image Segmentation
      </h1>
      <form onSubmit={handleSubmit} style={{ textAlign: 'center', marginBottom: '20px' }}>
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          style={{ marginBottom: '10px' }}
        />
        <br />
        <button
          type="submit"
          disabled={!selectedFile || loading}
          style={{
            padding: '10px 20px',
            fontSize: '16px',
            cursor: 'pointer',
            backgroundColor: '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '5px'
          }}
        >
          {loading ? 'Processing...' : 'Segment Image'}
        </button>
      </form>
      {imageUrl && (
        <div style={{ display: 'flex', justifyContent: 'space-around', gap: '20px' }}>
          <div style={{ textAlign: 'center' }}>
            <h3>Original Image</h3>
            <img
              src={imageUrl}
              alt="Original"
              style={{ maxWidth: '300px', borderRadius: '8px' }}
            />
          </div>
          {predictionUrl && (
            <div style={{ textAlign: 'center' }}>
              <h3>Segmented Image</h3>
              <img
                src={predictionUrl}
                alt="Prediction"
                style={{ maxWidth: '300px', borderRadius: '8px' }}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ImageUploader;
