import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { segmentImage } from '../api/unetApi';

const SegmentationViewer = () => {
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const onDrop = useCallback(async (acceptedFiles) => {
    if (acceptedFiles.length === 0) return;
    
    const file = acceptedFiles[0];
    setImage(file);
    setImagePreview(URL.createObjectURL(file));
    setResult(null);
    setError(null);
    setLoading(true);

    try {
      const data = await segmentImage(file);
      setResult(data);
    } catch (err) {
      setError(err.message || 'Failed to segment image');
    } finally {
      setLoading(false);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png']
    },
    multiple: false
  });

  const chartData = result ? [
    { name: 'Pet', value: result.class_distribution.Pet, color: '#ef4444' },
    { name: 'Background', value: result.class_distribution.Background, color: '#22c55e' },
    { name: 'Border', value: result.class_distribution.Border, color: '#3b82f6' }
  ] : [];

  return (
    <div className="w-full max-w-5xl mx-auto p-6 space-y-8 text-white">
      {/* Upload Section */}
      <div 
        {...getRootProps()} 
        className={`border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all duration-300 backdrop-blur-md bg-white/5 shadow-xl ${
          isDragActive ? 'border-blue-500 bg-blue-500/10' : 'border-gray-500 hover:border-gray-400 hover:bg-white/10'
        }`}
      >
        <input {...getInputProps()} />
        <div className="flex flex-col items-center justify-center space-y-4">
          <svg className="w-16 h-16 text-gray-400 drop-shadow-lg" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
          </svg>
          <p className="text-xl font-semibold tracking-wide text-gray-200">
            {isDragActive ? 'Drop image here...' : 'Drag & drop image here, or click to select'}
          </p>
          <p className="text-sm text-gray-400 font-medium">Supports JPG, PNG</p>
        </div>
      </div>

      {error && (
        <div className="bg-red-500/20 border border-red-500/50 text-red-200 p-4 rounded-xl backdrop-blur-md flex items-center shadow-lg">
          <svg className="w-6 h-6 mr-3 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
          <p>{error}</p>
        </div>
      )}

      {loading && (
        <div className="flex flex-col items-center justify-center py-16 space-y-6">
          <div className="w-16 h-16 border-4 border-blue-500/30 border-t-blue-500 rounded-full animate-spin drop-shadow-[0_0_15px_rgba(59,130,246,0.5)]"></div>
          <p className="text-xl font-medium tracking-widest text-blue-400 animate-pulse">Segmenting...</p>
        </div>
      )}

      {result && !loading && imagePreview && (
        <div className="space-y-8 animate-in fade-in zoom-in duration-500 ease-out">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="flex flex-col items-center p-6 bg-white/5 backdrop-blur-xl rounded-3xl border border-white/10 shadow-2xl">
              <h3 className="text-xl font-semibold mb-6 text-gray-200 tracking-wide uppercase text-sm">Original Image</h3>
              <div className="w-[256px] h-[256px] rounded-2xl overflow-hidden shadow-black/60 shadow-2xl border border-white/20">
                <img src={imagePreview} alt="Original" className="w-full h-full object-cover hover:scale-105 transition-transform duration-500" />
              </div>
            </div>
            
            <div className="flex flex-col items-center p-6 bg-white/5 backdrop-blur-xl rounded-3xl border border-white/10 shadow-2xl">
              <h3 className="text-xl font-semibold mb-6 text-gray-200 tracking-wide uppercase text-sm">Segmentation Mask</h3>
              <div className="w-[256px] h-[256px] rounded-2xl overflow-hidden shadow-black/60 shadow-2xl border border-white/20 bg-gray-900">
                <img src={`data:image/png;base64,${result.mask_base64}`} alt="Mask" className="w-full h-full object-cover opacity-90 hover:opacity-100 transition-opacity duration-300" />
              </div>
            </div>
          </div>

          <div className="bg-white/5 backdrop-blur-xl rounded-3xl border border-white/10 p-8 shadow-2xl">
            <h3 className="text-2xl font-bold text-center mb-8 text-gray-100">Class Distribution</h3>
            
            <div className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={chartData}
                    cx="50%"
                    cy="50%"
                    innerRadius={80}
                    outerRadius={100}
                    paddingAngle={5}
                    dataKey="value"
                    stroke="none"
                  >
                    {chartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} style={{ filter: `drop-shadow(0px 0px 8px ${entry.color}80)` }} />
                    ))}
                  </Pie>
                  <Tooltip 
                    formatter={(value) => `${value.toFixed(2)}%`}
                    contentStyle={{ 
                      backgroundColor: 'rgba(15, 23, 42, 0.9)', 
                      border: '1px solid rgba(255,255,255,0.1)', 
                      borderRadius: '12px',
                      boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.5)'
                    }}
                    itemStyle={{ color: '#e2e8f0', fontWeight: '500' }}
                  />
                  <Legend 
                    verticalAlign="bottom" height={36}
                    formatter={(value, entry) => (
                      <span className="text-gray-200 font-medium ml-2 mr-4">
                        {value} <span className="text-gray-400">({entry.payload.value.toFixed(1)}%)</span>
                      </span>
                    )}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div className="mt-8 flex justify-center animate-bounce">
              <div className="inline-flex items-center px-8 py-3 rounded-full bg-gradient-to-r from-blue-600/20 to-purple-600/20 border border-blue-500/30 shadow-[0_0_20px_rgba(59,130,246,0.2)]">
                <span className="text-blue-300 font-semibold uppercase tracking-widest text-xs mr-3">Dominant Class</span>
                <span className="text-white font-bold text-xl drop-shadow-md">{result.dominant_class}</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default SegmentationViewer;
