import { useState } from 'react';
import { segmentImage } from '../api/unetApi';

/**
 * ImageUploader – legacy fallback upload component.
 * Uses the /api/segment endpoint via unetApi.js.
 * The main segmentation UI is in SegmentationViewer.jsx.
 */
const ImageUploader = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [maskUrl, setMaskUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setImageUrl(URL.createObjectURL(file));
      setMaskUrl(null);
      setError(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    try {
      const data = await segmentImage(selectedFile);
      // Convert base64 mask to an object URL for display
      const byteString = atob(data.mask_base64);
      const arrayBuffer = new ArrayBuffer(byteString.length);
      const uint8Array = new Uint8Array(arrayBuffer);
      for (let i = 0; i < byteString.length; i++) {
        uint8Array[i] = byteString.charCodeAt(i);
      }
      const blob = new Blob([uint8Array], { type: 'image/png' });
      setMaskUrl(URL.createObjectURL(blob));
    } catch (err) {
      console.error('Error segmenting image:', err);
      setError(err.message || 'Segmentation failed');
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
            borderRadius: '5px',
          }}
        >
          {loading ? 'Processing...' : 'Segment Image'}
        </button>
      </form>

      {error && (
        <p style={{ color: 'red', textAlign: 'center' }}>{error}</p>
      )}

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
          {maskUrl && (
            <div style={{ textAlign: 'center' }}>
              <h3>Segmented Mask</h3>
              <img
                src={maskUrl}
                alt="Segmentation Mask"
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
